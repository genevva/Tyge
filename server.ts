/**
 * ============================================================================
 * HAI (High-level Agent Interface) - TypeScript Single File Edition
 * ============================================================================
 * 
 * 将 Anthropic 的 claude-code CLI 封装为标准的 Messages API 网关
 * 
 * 核心特性：
 * - 完全兼容 Anthropic Messages API 格式
 * - 支持流式 (SSE) 和非流式响应
 * - 历史对话重放（History Replay）架构
 * - 支持 Thinking、工具调用、图片等高级功能
 * 
 * Author: Max (Refactored from Python version)
 * Version: 2.0.1-ts (Fixed settingSources issue)
 * ============================================================================
 */

import express, { Request, Response } from 'express';
import cors from 'cors';
import { randomUUID } from 'crypto';
import {
  query,
  type Options,
  type SDKUserMessage,
  type SDKMessage,
  type SDKAssistantMessage,
  type SDKResultMessage,
} from '@anthropic-ai/claude-agent-sdk';

// ============================================================================
// 1. 配置 (Configuration)
// ============================================================================

const CONFIG = {
  // Claude CLI 可执行文件路径（留空则自动探测）
  CLAUDE_CLI_PATH: process.env.CLAUDE_CLI_PATH || '',

  // HTTP 服务配置
  API_HOST: process.env.API_HOST || '0.0.0.0',
  API_PORT: Number(process.env.API_PORT || '8000'),

  // 默认工作目录
  DEFAULT_CWD: process.env.DEFAULT_CWD || '/tmp/',

  // 默认允许的工具列表
  DEFAULT_ALLOWED_TOOLS: (process.env.DEFAULT_ALLOWED_TOOLS || 'WebSearch')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean),

  // 默认模型
  DEFAULT_MODEL: process.env.DEFAULT_MODEL || 'claude-sonnet-4-5',

  // Thinking 配置
  DEFAULT_MAX_THINKING_TOKENS: Number(
    process.env.DEFAULT_MAX_THINKING_TOKENS || '8000'
  ),
  ENABLE_THINKING_BY_DEFAULT:
    (process.env.ENABLE_THINKING_BY_DEFAULT || 'true').toLowerCase() === 'true',

  // 权限模式
  DEFAULT_PERMISSION_MODE:
    (process.env.DEFAULT_PERMISSION_MODE as
      | 'default'
      | 'acceptEdits'
      | 'bypassPermissions'
      | 'plan') || 'acceptEdits',

  // 最大轮次
  DEFAULT_MAX_TURNS: Number(process.env.DEFAULT_MAX_TURNS || '99999'),

  // 设置源配置（新增）
  // 可选值：'user', 'project', 'local' 的组合，用逗号分隔
  // 留空表示不加载任何文件系统设置（推荐，完全由代码控制）
  SETTING_SOURCES: (process.env.SETTING_SOURCES || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean) as ('user' | 'project' | 'local')[],

  // 调试模式
  DEBUG: (process.env.DEBUG || 'false').toLowerCase() === 'true',
} as const;

// ============================================================================
// 2. 类型定义 (Type Definitions)
// ============================================================================

// 消息角色
type Role = 'user' | 'assistant';

// 内容块基类
interface ContentBlockBase {
  type: string;
}

// 文本内容
interface TextContent extends ContentBlockBase {
  type: 'text';
  text: string;
}

// 思考内容（Extended Thinking）
interface ThinkingContent extends ContentBlockBase {
  type: 'thinking';
  thinking: string;
}

// 图片内容
interface ImageContent extends ContentBlockBase {
  type: 'image';
  source: {
    type: 'base64' | 'url' | string;
    media_type?: string;
    data?: string;
    url?: string;
  };
}

// 工具使用
interface ToolUseContent extends ContentBlockBase {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

// 工具结果
interface ToolResultContent extends ContentBlockBase {
  type: 'tool_result';
  tool_use_id: string;
  content?: string | Array<Record<string, unknown>>;
  is_error?: boolean;
}

// 内容块联合类型
type ContentBlock =
  | TextContent
  | ThinkingContent
  | ImageContent
  | ToolUseContent
  | ToolResultContent;

// 消息对象
interface Message {
  role: Role;
  content: string | ContentBlock[];
}

// Messages API 请求体（保持 snake_case 以兼容 Anthropic API）
interface MessagesRequest {
  model: string;
  messages: Message[];
  max_tokens?: number;
  system?: string | any[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
  metadata?: Record<string, unknown>;

  // Claude Code 特有选项
  tools?: string[];
  max_turns?: number;
  permission_mode?: string;
  cwd?: string;

  // Thinking tokens
  max_thinking_tokens?: number;
}

// Token 使用统计
interface Usage {
  input_tokens: number;
  output_tokens: number;
}

// Messages API 响应体
interface MessagesResponse {
  id: string;
  type: 'message';
  role: 'assistant';
  content: Array<TextContent | ThinkingContent | ToolUseContent>;
  model: string;
  stop_reason?: string | null;
  stop_sequence?: string | null;
  usage: Usage;
}

// ============================================================================
// 3. 历史重放转换器 (History Replay Converter)
// ============================================================================

/**
 * 将 Messages API 格式的对话历史转换为 SDK 的 Streaming Input 格式
 * 
 * 核心策略：
 * 1. 将历史消息封装在 <conversation_history> XML 标签中
 * 2. 将当前消息封装在 <current_question> XML 标签中
 * 3. 通过单次 AsyncIterable<SDKUserMessage> 喂给 Agent SDK
 */
class HistoryReplayConverter {
  /**
   * 主入口：将完整的 messages[] 转换为 SDK 可接受的流式输入
   */
  static messagesToStreamingInput(
    messages: Message[],
    sessionId: string
  ): AsyncIterable<SDKUserMessage> {
    return (async function* () {
      if (!messages.length) return;

      // 分离：历史消息 vs 当前消息
      const history = messages.slice(0, -1);
      const current = messages[messages.length - 1];

      if (current.role !== 'user') {
        throw new Error('最后一条消息必须是 user 角色');
      }

      const contentBlocks: any[] = [];

      // 1) 构建历史对话上下文
      if (history.length > 0) {
        const historyText =
          HistoryReplayConverter.buildConversationHistory(history);
        contentBlocks.push({
          type: 'text',
          text: historyText,
        });
      }

      // 2) 处理当前问题（可能包含图片等多模态内容）
      const currentContent =
        HistoryReplayConverter.processCurrentMessage(current);
      contentBlocks.push(...currentContent);

      // 3) 生成 SDK 格式的 UserMessage
      const userMessage: SDKUserMessage = {
        type: 'user',
        session_id: sessionId,
        parent_tool_use_id: null,
        message: {
          role: 'user',
          content: contentBlocks,
        },
      };

      yield userMessage;
    })();
  }

  /**
   * 构建 <conversation_history> 块
   */
  static buildConversationHistory(messages: Message[]): string {
    const lines: string[] = [];
    lines.push('<conversation_history>');
    lines.push(
      'This is the previous conversation for context. You should be aware of it, ' +
        'but respond ONLY to the <current_question> below.'
    );
    lines.push('');

    let i = 0;
    while (i < messages.length) {
      const msg = messages[i];

      if (msg.role === 'user') {
        const userText = this.extractTextContent(msg);
        lines.push('<user>');
        lines.push(this.escapeXml(userText));
        lines.push('</user>');

        // 检查下一条是否是 assistant 响应
        if (i + 1 < messages.length && messages[i + 1].role === 'assistant') {
          const asst = messages[i + 1];
          const asstText = this.extractTextContent(asst);
          lines.push('<assistant>');
          lines.push(this.escapeXml(asstText));
          lines.push('</assistant>');
          i += 2;
        } else {
          i += 1;
        }
      } else if (msg.role === 'assistant') {
        // 单独的 assistant 消息（可能不规范，但容错处理）
        const asstText = this.extractTextContent(msg);
        lines.push('<assistant>');
        lines.push(this.escapeXml(asstText));
        lines.push('</assistant>');
        i += 1;
      } else {
        i += 1;
      }
    }

    lines.push('</conversation_history>');
    lines.push('');
    return lines.join('\n');
  }

  /**
   * 处理当前消息，提取文本和多模态内容
   */
  static processCurrentMessage(msg: Message): any[] {
    const contentBlocks: any[] = [];

    if (typeof msg.content === 'string') {
      // 简单字符串模式
      contentBlocks.push({
        type: 'text',
        text: `<current_question>\n${msg.content}\n</current_question>`,
      });
      return contentBlocks;
    }

    // 结构化内容模式
    const textParts: string[] = [];
    const images: ImageContent[] = [];
    const toolUses: ToolUseContent[] = [];
    const toolResults: ToolResultContent[] = [];

    for (const block of msg.content) {
      switch (block.type) {
        case 'text':
          textParts.push((block as TextContent).text);
          break;
        case 'image':
          images.push(block as ImageContent);
          break;
        case 'tool_use':
          toolUses.push(block as ToolUseContent);
          break;
        case 'tool_result':
          toolResults.push(block as ToolResultContent);
          break;
      }
    }

    const currentQuestionParts: string[] = [];

    if (textParts.length > 0) {
      currentQuestionParts.push(textParts.join('\n'));
    }

    // 添加工具调用的元信息
    for (const tool of toolUses) {
      currentQuestionParts.push(
        `[Previously used tool: ${tool.name} with id=${tool.id}]`
      );
    }

    // 添加工具结果的摘要
    for (const result of toolResults) {
      let resultText = '';
      if (typeof result.content === 'string') {
        resultText = result.content;
      } else if (Array.isArray(result.content)) {
        resultText = JSON.stringify(result.content);
      }
      currentQuestionParts.push(
        `[Tool result for ${result.tool_use_id}]: ${resultText.slice(0, 200)}`
      );
    }

    // 组装文本块
    if (currentQuestionParts.length > 0) {
      const fullText = currentQuestionParts.join('\n');
      contentBlocks.push({
        type: 'text',
        text: `<current_question>\n${fullText}\n</current_question>`,
      });
    }

    // 添加图片块（保持原始格式）
    for (const img of images) {
      contentBlocks.push({
        type: 'image',
        source: {
          type: img.source.type,
          media_type: img.source.media_type || 'image/jpeg',
          data: img.source.type === 'base64' ? img.source.data : undefined,
          url: img.source.type === 'url' ? img.source.url : undefined,
        },
      });
    }

    return contentBlocks;
  }

  /**
   * 从消息中提取纯文本内容（用于历史回放）
   */
  static extractTextContent(msg: Message): string {
    if (typeof msg.content === 'string') return msg.content;

    const parts: string[] = [];
    for (const block of msg.content) {
      switch (block.type) {
        case 'text':
          parts.push((block as TextContent).text);
          break;
        case 'tool_use': {
          const t = block as ToolUseContent;
          let desc = `[Used tool: ${t.name}`;
          if (t.input) {
            const preview = JSON.stringify(t.input).slice(0, 100);
            desc += ` with input: ${preview}`;
          }
          desc += ']';
          parts.push(desc);
          break;
        }
        case 'tool_result': {
          const tr = block as ToolResultContent;
          if (typeof tr.content === 'string') {
            parts.push(`[Tool result: ${tr.content.slice(0, 200)}...]`);
          } else if (Array.isArray(tr.content)) {
            parts.push(`[Tool result: ${tr.content.length} items]`);
          }
          break;
        }
        case 'image':
          parts.push('[Image attached]');
          break;
      }
    }
    return parts.join('\n');
  }

  /**
   * XML 转义
   */
  static escapeXml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }
}

// ============================================================================
// 4. 消息格式转换器 (Message Converter)
// ============================================================================

/**
 * 处理非流式模式下的消息聚合和响应构建
 */
class MessageConverter {
  /**
   * 将 SDK 的 AssistantMessage 转换为 API 内容格式
   */
  static sdkToApiContent(sdkMessage: SDKAssistantMessage): any[] {
    const apiContent: any[] = [];

    const contentBlocks = (sdkMessage as any).message.content as any[];

    for (const block of contentBlocks) {
      if (block.type === 'text') {
        apiContent.push({
          type: 'text',
          text: block.text,
        });
      } else if (block.type === 'thinking') {
        apiContent.push({
          type: 'thinking',
          thinking: block.thinking,
        });
      } else if (block.type === 'tool_use') {
        apiContent.push({
          type: 'tool_use',
          id: block.id,
          name: block.name,
          input: block.input,
        });
      }
    }

    return apiContent;
  }

  /**
   * 构建非流式响应对象
   */
  static buildNonStreamingResponse(
    assistantMessages: SDKMessage[],
    result: SDKResultMessage,
    model: string,
    requestId: string
  ): MessagesResponse {
    const allContent: any[] = [];

    // 聚合所有 assistant 消息的内容
    for (const msg of assistantMessages) {
      if (msg.type !== 'assistant') continue;
      const content = MessageConverter.sdkToApiContent(
        msg as SDKAssistantMessage
      );
      allContent.push(...content);
    }

    // 空内容兜底
    if (allContent.length === 0) {
      allContent.push({ type: 'text', text: '' });
    }

    // 映射停止原因
    let stopReason = 'end_turn';
    if (result.subtype === 'error_max_turns') {
      stopReason = 'max_tokens';
    } else if (result.is_error) {
      stopReason = 'error';
    }

    // 构建 usage
    const usage: Usage = {
      input_tokens: result.usage.input_tokens || 0,
      output_tokens: result.usage.output_tokens || 0,
    };

    return {
      id: requestId,
      type: 'message',
      role: 'assistant',
      content: allContent,
      model,
      stop_reason: stopReason,
      stop_sequence: null,
      usage,
    };
  }
}

// ============================================================================
// 5. SDK Options 构建器 (SDK Options Builder)
// ============================================================================

/**
 * 根据 API 请求参数构建 Agent SDK 的 Options
 */
function buildSdkOptions(reqBody: MessagesRequest, isStream: boolean): Options {
  const options: Options = {
    cwd: reqBody.cwd || CONFIG.DEFAULT_CWD,
    allowedTools: CONFIG.DEFAULT_ALLOWED_TOOLS,
    permissionMode: (reqBody.permission_mode as any) || CONFIG.DEFAULT_PERMISSION_MODE,
    maxTurns: reqBody.max_turns || CONFIG.DEFAULT_MAX_TURNS,
    model: reqBody.model === CONFIG.DEFAULT_MODEL ? undefined : reqBody.model,
    includePartialMessages: isStream,
    env: { ...process.env } as Record<string, string>,
    extraArgs: {},
    
    // 🔧 关键修复：明确设置 settingSources
    // 空数组 = 不加载任何文件系统设置（推荐）
    // ['project'] = 只加载项目设置（会读取 CLAUDE.md）
    // ['user', 'project', 'local'] = 加载所有设置源
    settingSources: CONFIG.SETTING_SOURCES.length > 0 ? CONFIG.SETTING_SOURCES : [],
  };

  // CLI 路径（如果配置了）
  if (CONFIG.CLAUDE_CLI_PATH) {
    (options as any).pathToClaudeCodeExecutable = CONFIG.CLAUDE_CLI_PATH;
  }

  // System Prompt 提取
  let systemInstruction: string | undefined;
  if (typeof reqBody.system === 'string') {
    systemInstruction = reqBody.system;
  } else if (Array.isArray(reqBody.system)) {
    const texts = reqBody.system
      .filter((b) => b?.type === 'text')
      .map((b) => b.text || '')
      .join('\n');
    systemInstruction = texts;
  }

  if (systemInstruction) {
    options.systemPrompt = systemInstruction;
  }

  // Thinking Tokens 配置
  if (typeof reqBody.max_thinking_tokens === 'number') {
    options.maxThinkingTokens = reqBody.max_thinking_tokens;
  } else if (CONFIG.ENABLE_THINKING_BY_DEFAULT) {
    options.maxThinkingTokens = CONFIG.DEFAULT_MAX_THINKING_TOKENS;
  }

  // 环境变量注入
  if (!options.env) options.env = {};

  if (options.maxThinkingTokens) {
    options.env['MAX_THINKING_TOKENS'] = String(options.maxThinkingTokens);
  }

  if (reqBody.temperature != null) {
    options.env['ANTHROPIC_TEMPERATURE'] = String(reqBody.temperature);
  }
  if (reqBody.top_p != null) {
    options.env['ANTHROPIC_TOP_P'] = String(reqBody.top_p);
  }
  if (reqBody.top_k != null) {
    options.env['ANTHROPIC_TOP_K'] = String(reqBody.top_k);
  }

  // Debug 模式
  if (CONFIG.DEBUG) {
    options.extraArgs = {
      ...(options.extraArgs || {}),
      'debug-to-stderr': null,
    } as any;
  }

  return options;
}

// ============================================================================
// 6. 流式处理器 (Streaming Handler)
// ============================================================================

/**
 * 处理流式请求，通过 SSE 推送事件
 */
async function handleStreamingWithHistoryReplay(
  messages: Message[],
  options: Options,
  model: string,
  requestId: string,
  req: Request,
  res: Response
): Promise<void> {
  const abortController = new AbortController();
  options.abortController = abortController;

  // 设置 SSE 响应头
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Request-ID', requestId);

  if (CONFIG.DEBUG) {
    console.log(`🔄 开始流式处理 [${requestId}]`);
    console.log(`   重放 ${messages.length} 条历史消息...`);
  }

  // 客户端断开时取消请求
  req.on('close', () => {
    abortController.abort();
  });

  const sessionId = requestId;
  const promptStream = HistoryReplayConverter.messagesToStreamingInput(
    messages,
    sessionId
  );

  const q = query({
    prompt: promptStream,
    options,
  });

  try {
    for await (const sdkMsg of q as AsyncIterable<SDKMessage>) {
      // 流式事件：直接透传 SDK 的 RawMessageStreamEvent
      if (sdkMsg.type === 'stream_event') {
        const event = (sdkMsg as any).event;
        const eventType = event.type;
        res.write(`event: ${eventType}\ndata: ${JSON.stringify(event)}\n\n`);
      }
      // 其他类型（assistant / result）主要用于统计，这里不额外处理
    }

    if (CONFIG.DEBUG) {
      console.log(`✅ 流式响应完成 [${requestId}]`);
    }
    res.end();
  } catch (err: any) {
    if (CONFIG.DEBUG) {
      console.error('❌ 流式处理错误', err);
    }
    const errorEvent = {
      type: 'error',
      error: {
        type: 'api_error',
        message: err?.message || String(err),
      },
    };
    res.write(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`);
    res.end();
  }
}

// ============================================================================
// 7. 非流式处理器 (Non-Streaming Handler)
// ============================================================================

/**
 * 处理非流式请求，返回完整的 JSON 响应
 */
async function handleNonStreamingWithHistoryReplay(
  messages: Message[],
  options: Options,
  model: string,
  requestId: string,
  res: Response
): Promise<void> {
  if (CONFIG.DEBUG) {
    console.log(`🔄 开始非流式处理 [${requestId}]`);
    console.log(`   重放 ${messages.length} 条历史消息...`);
  }

  const sessionId = requestId;
  const promptStream = HistoryReplayConverter.messagesToStreamingInput(
    messages,
    sessionId
  );

  const q = query({
    prompt: promptStream,
    options,
  });

  const assistantMessages: SDKMessage[] = [];
  let resultMessage: SDKResultMessage | null = null;

  // 收集所有消息
  for await (const sdkMsg of q as AsyncIterable<SDKMessage>) {
    if (sdkMsg.type === 'assistant') {
      assistantMessages.push(sdkMsg);
    } else if (sdkMsg.type === 'result') {
      resultMessage = sdkMsg as SDKResultMessage;
    }
  }

  if (!resultMessage) {
    throw new Error('未收到结果消息');
  }

  // 构建响应
  const response = MessageConverter.buildNonStreamingResponse(
    assistantMessages,
    resultMessage,
    model,
    requestId
  );

  if (CONFIG.DEBUG) {
    console.log(`✅ 非流式响应完成 [${requestId}]`);
    console.log(
      `   输入 tokens: ${response.usage.input_tokens}, 输出 tokens: ${response.usage.output_tokens}`
    );
    if (resultMessage.total_cost_usd) {
      console.log(`   总成本: $${resultMessage.total_cost_usd.toFixed(4)}`);
    }
  }

  res.setHeader('X-Request-ID', requestId);
  res.json(response);
}

// ============================================================================
// 8. Express 应用 (Express Application)
// ============================================================================

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// 全局统计
let requestCount = 0;
let successCount = 0;
let errorCount = 0;

// ---- 服务状态端点 ----

app.get('/', (_req, res) => {
  res.json({
    service: 'Claude Messages API Gateway (TypeScript)',
    version: '2.0.1-ts',
    status: 'running',
    features: {
      streaming: true,
      history_replay: true,
      thinking: CONFIG.ENABLE_THINKING_BY_DEFAULT,
      tools: true,
      images: true,
    },
    configuration: {
      setting_sources: CONFIG.SETTING_SOURCES.length > 0 
        ? CONFIG.SETTING_SOURCES 
        : 'none (code-only)',
      permission_mode: CONFIG.DEFAULT_PERMISSION_MODE,
      max_turns: CONFIG.DEFAULT_MAX_TURNS,
    },
    statistics: {
      total_requests: requestCount,
      successful: successCount,
      failed: errorCount,
    },
  });
});

app.get('/health', (_req, res) => {
  res.json({
    status: 'healthy',
    timestamp: Date.now() / 1000,
  });
});

// ---- Messages API 端点 ----

app.post('/v1/messages', async (req: Request, res: Response) => {
  requestCount += 1;
  const requestId = `msg_${randomUUID().replace(/-/g, '').slice(0, 24)}`;

  const body = req.body as MessagesRequest;

  if (CONFIG.DEBUG) {
    console.log('\n' + '='.repeat(70));
    console.log(`📨 新请求 [${requestId}]`);
    console.log(`   模型: ${body.model}`);
    console.log(`   消息数: ${body.messages?.length || 0}`);
    console.log(`   流式: ${body.stream}`);
    if (body.max_thinking_tokens) {
      console.log(`   Thinking Tokens: ${body.max_thinking_tokens}`);
    }
    if (body.tools) {
      console.log(`   自定义工具: ${body.tools.join(', ')}`);
    }
    console.log('='.repeat(70) + '\n');
  }

  try {
    // 参数校验
    if (!body.messages || body.messages.length === 0) {
      errorCount += 1;
      return res.status(400).json({
        type: 'invalid_request_error',
        message: 'messages 不能为空',
      });
    }

    if (body.messages[body.messages.length - 1].role !== 'user') {
      errorCount += 1;
      return res.status(400).json({
        type: 'invalid_request_error',
        message: '最后一条消息必须是 user 角色',
      });
    }

    const options = buildSdkOptions(body, !!body.stream);

    if (body.stream) {
      await handleStreamingWithHistoryReplay(
        body.messages,
        options,
        body.model,
        requestId,
        req,
        res
      );
      successCount += 1;
    } else {
      await handleNonStreamingWithHistoryReplay(
        body.messages,
        options,
        body.model,
        requestId,
        res
      );
      successCount += 1;
    }
  } catch (err: any) {
    errorCount += 1;

    if (CONFIG.DEBUG) {
      console.error('\n' + '='.repeat(70));
      console.error(`❌ 错误 [${requestId}]`);
      console.error('='.repeat(70));
      console.error(err);
      console.error('='.repeat(70) + '\n');
    }

    res.status(500).json({
      type: 'api_error',
      message: err?.message || String(err),
    });
  }
});

// ---- Token 计数端点 ----

app.post('/v1/messages/count-tokens', (req: Request, res: Response) => {
  const data = req.body as any;
  const messages = data.messages || [];
  const system = data.system || '';

  let totalChars = String(system).length;

  for (const msg of messages) {
    const content = msg.content;
    if (typeof content === 'string') {
      totalChars += content.length;
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (block.type === 'text') {
          totalChars += (block.text || '').length;
        }
      }
    }
  }

  // 粗略估算（中英文混合，每字符约 0.5 token）
  const estimatedTokens = Math.floor(totalChars * 0.5);

  res.json({ input_tokens: estimatedTokens });
});

// ============================================================================
// 9. 服务器启动 (Server Startup)
// ============================================================================

app.listen(CONFIG.API_PORT, CONFIG.API_HOST, () => {
  console.log('\n' + '='.repeat(70));
  console.log('🚀 Claude Messages API Gateway (TypeScript) 启动完成');
  console.log('='.repeat(70));
  console.log(`📍 监听地址: http://${CONFIG.API_HOST}:${CONFIG.API_PORT}`);
  console.log(`📁 默认工作目录: ${CONFIG.DEFAULT_CWD}`);
  console.log(`🛠  默认工具: ${CONFIG.DEFAULT_ALLOWED_TOOLS.join(', ')}`);
  console.log(`🤖 默认模型: ${CONFIG.DEFAULT_MODEL}`);
  console.log(`🧠 Thinking 默认启用: ${CONFIG.ENABLE_THINKING_BY_DEFAULT}`);
  if (CONFIG.ENABLE_THINKING_BY_DEFAULT) {
    console.log(`💭 默认 Thinking Tokens: ${CONFIG.DEFAULT_MAX_THINKING_TOKENS}`);
  }
  console.log(`🔐 权限模式: ${CONFIG.DEFAULT_PERMISSION_MODE}`);
  console.log(`⚙️  设置源: ${CONFIG.SETTING_SOURCES.length > 0 ? CONFIG.SETTING_SOURCES.join(', ') : '无（纯代码控制）'}`);
  console.log(`🔄 最大轮次: ${CONFIG.DEFAULT_MAX_TURNS}`);
  console.log(`🐞 调试模式: ${CONFIG.DEBUG}`);
  console.log('='.repeat(70));
  console.log('✅ 服务已就绪，等待请求...\n');
});
