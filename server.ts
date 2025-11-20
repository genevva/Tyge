/**
 * HAI (High-level Agent Interface)
 * Claude Messages API Gateway - TypeScript Edition
 * 
 * 单文件完整实现，基于官方 @anthropic-ai/claude-agent-sdk
 */

import 'dotenv/config';
import Fastify, { FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import { query, type SDKUserMessage, type SDKMessage, type SDKAssistantMessage, type SDKResultMessage, type SDKPartialAssistantMessage, type Options } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import { randomUUID } from 'crypto';

// ============================================================================
// 配置管理
// ============================================================================

const CONFIG = {
  // 服务器配置
  server: {
    host: process.env.API_HOST || '0.0.0.0',
    port: parseInt(process.env.API_PORT || '8000', 10),
  },

  // Claude Code 配置
  claude: {
    cliPath: process.env.CLAUDE_CLI_PATH || 'claude',
    defaultCwd: process.env.DEFAULT_CWD || '/tmp',
    defaultModel: process.env.DEFAULT_MODEL || 'claude-sonnet-4-5',
    defaultAllowedTools: process.env.DEFAULT_ALLOWED_TOOLS?.split(',') || ['WebSearch'],
    permissionMode: (process.env.DEFAULT_PERMISSION_MODE || 'acceptEdits') as 'default' | 'acceptEdits' | 'bypassPermissions' | 'plan',
    maxTurns: parseInt(process.env.DEFAULT_MAX_TURNS || '99999', 10),
  },

  // Thinking 配置
  thinking: {
    enabled: process.env.ENABLE_THINKING_BY_DEFAULT !== 'false',
    defaultMaxTokens: parseInt(process.env.DEFAULT_MAX_THINKING_TOKENS || '8000', 10),
  },

  // 调试
  debug: process.env.DEBUG === 'true',
} as const;

// ============================================================================
// 类型定义
// ============================================================================

// Content Blocks
const TextContentSchema = z.object({
  type: z.literal('text'),
  text: z.string(),
});

const ImageContentSchema = z.object({
  type: z.literal('image'),
  source: z.object({
    type: z.enum(['base64', 'url']),
    media_type: z.string().optional(),
    data: z.string().optional(),
    url: z.string().optional(),
  }),
});

const ToolUseContentSchema = z.object({
  type: z.literal('tool_use'),
  id: z.string(),
  name: z.string(),
  input: z.record(z.any()),
});

const ToolResultContentSchema = z.object({
  type: z.literal('tool_result'),
  tool_use_id: z.string(),
  content: z.union([z.string(), z.array(z.any())]).optional(),
  is_error: z.boolean().optional(),
});

const ThinkingContentSchema = z.object({
  type: z.literal('thinking'),
  thinking: z.string(),
});

const ContentBlockSchema = z.union([
  TextContentSchema,
  ImageContentSchema,
  ToolUseContentSchema,
  ToolResultContentSchema,
  ThinkingContentSchema,
]);

// Messages
const MessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.union([z.string(), z.array(ContentBlockSchema)]),
});

// API Request
const MessagesRequestSchema = z.object({
  model: z.string(),
  messages: z.array(MessageSchema),
  max_tokens: z.number().default(4096),
  system: z.union([z.string(), z.array(z.any())]).optional(),
  stream: z.boolean().optional().default(false),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  top_k: z.number().optional(),
  stop_sequences: z.array(z.string()).optional(),
  
  // Claude Code 特定
  tools: z.array(z.string()).optional(),
  max_turns: z.number().optional(),
  permission_mode: z.enum(['default', 'acceptEdits', 'bypassPermissions', 'plan']).optional(),
  cwd: z.string().optional(),
  max_thinking_tokens: z.number().optional(),
});

type Message = z.infer<typeof MessageSchema>;
type MessagesRequest = z.infer<typeof MessagesRequestSchema>;
type ContentBlock = z.infer<typeof ContentBlockSchema>;

// ============================================================================
// 统计数据
// ============================================================================

const statistics = {
  totalRequests: 0,
  successfulRequests: 0,
  failedRequests: 0,
};

// ============================================================================
// History Replay 转换器
// ============================================================================

class HistoryReplayConverter {
  /**
   * 将 Messages API 格式转换为 SDK streaming input
   */
  static async *convertToStreamingInput(
    messages: Message[],
    sessionId: string,
  ): AsyncGenerator<SDKUserMessage> {
    if (messages.length === 0) return;

    const history = messages.slice(0, -1);
    const current = messages[messages.length - 1];

    if (current.role !== 'user') {
      throw new Error('Last message must be a user message');
    }

    const contentBlocks: any[] = [];

    // 1. 添加历史对话
    if (history.length > 0) {
      contentBlocks.push({
        type: 'text',
        text: this.buildConversationHistory(history),
      });
    }

    // 2. 添加当前问题
    contentBlocks.push(...this.processCurrentMessage(current));

    // 3. 生成单条 user 消息
    yield {
      type: 'user',
      message: {
        role: 'user',
        content: contentBlocks,
      },
      parent_tool_use_id: null,
      session_id: sessionId,
    };
  }

  /**
   * 构建 XML 格式的历史对话
   */
  private static buildConversationHistory(messages: Message[]): string {
    const lines = [
      '<conversation_history>',
      'This is the previous conversation for context. You should be aware of it, but respond ONLY to the <current_question> below.',
      '',
    ];

    let i = 0;
    while (i < messages.length) {
      const msg = messages[i];

      if (msg.role === 'user') {
        lines.push('<user>');
        lines.push(this.escapeXml(this.extractTextContent(msg)));
        lines.push('</user>');

        if (i + 1 < messages.length && messages[i + 1].role === 'assistant') {
          const asst = messages[i + 1];
          lines.push('<assistant>');
          lines.push(this.escapeXml(this.extractTextContent(asst)));
          lines.push('</assistant>');
          i += 2;
        } else {
          i += 1;
        }
      } else if (msg.role === 'assistant') {
        lines.push('<assistant>');
        lines.push(this.escapeXml(this.extractTextContent(msg)));
        lines.push('</assistant>');
        i += 1;
      } else {
        i += 1;
      }
    }

    lines.push('</conversation_history>', '');
    return lines.join('\n');
  }

  /**
   * 处理当前用户消息
   */
  private static processCurrentMessage(msg: Message): any[] {
    const contentBlocks: any[] = [];

    if (typeof msg.content === 'string') {
      contentBlocks.push({
        type: 'text',
        text: `<current_question>\n${msg.content}\n</current_question>`,
      });
    } else {
      const textParts: string[] = [];
      const images: any[] = [];

      for (const block of msg.content) {
        if (block.type === 'text') {
          textParts.push(block.text);
        } else if (block.type === 'image') {
          images.push(block);
        } else if (block.type === 'tool_use') {
          textParts.push(`[Previously used tool: ${block.name} with id=${block.id}]`);
        } else if (block.type === 'tool_result') {
          const resultText = typeof block.content === 'string' ? block.content : JSON.stringify(block.content);
          textParts.push(`[Tool result for ${block.tool_use_id}]: ${resultText.slice(0, 200)}`);
        }
      }

      if (textParts.length > 0) {
        contentBlocks.push({
          type: 'text',
          text: `<current_question>\n${textParts.join('\n')}\n</current_question>`,
        });
      }

      for (const img of images) {
        contentBlocks.push({ type: 'image', source: img.source });
      }
    }

    return contentBlocks;
  }

  /**
   * 从消息中提取纯文本
   */
  private static extractTextContent(msg: Message): string {
    if (typeof msg.content === 'string') return msg.content;

    const parts: string[] = [];
    for (const block of msg.content) {
      if (block.type === 'text') {
        parts.push(block.text);
      } else if (block.type === 'tool_use') {
        parts.push(`[Used tool: ${block.name}]`);
      } else if (block.type === 'tool_result') {
        const preview = typeof block.content === 'string' ? block.content.slice(0, 200) : '[Tool result]';
        parts.push(`[Tool result: ${preview}]`);
      } else if (block.type === 'image') {
        parts.push('[Image attached]');
      }
    }
    return parts.join('\n');
  }

  /**
   * XML 转义
   */
  private static escapeXml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }
}

// ============================================================================
// 流式转换器
// ============================================================================

class StreamConverter {
  private messageStarted = false;
  private contentBlocksStarted = new Set<number>();
  private totalInputTokens = 0;
  private totalOutputTokens = 0;

  constructor(
    private model: string,
    private requestId: string,
  ) {}

  /**
   * 转换 SDK 流为 SSE 格式
   */
  async *convert(sdkStream: AsyncGenerator<SDKMessage>): AsyncGenerator<string> {
    try {
      for await (const message of sdkStream) {
        if (message.type === 'stream_event') {
          yield* this.handleStreamEvent(message as SDKPartialAssistantMessage);
        } else if (message.type === 'result') {
          const result = message as SDKResultMessage;
          if (result.usage) {
            this.totalInputTokens = result.usage.input_tokens;
            this.totalOutputTokens = result.usage.output_tokens;
          }
        }
      }
    } catch (error) {
      yield this.formatEvent('error', {
        type: 'error',
        error: {
          type: 'api_error',
          message: error instanceof Error ? error.message : String(error),
        },
      });
    }
  }

  /**
   * 处理流式事件
   */
  private async *handleStreamEvent(sdkEvent: SDKPartialAssistantMessage): AsyncGenerator<string> {
    const event = sdkEvent.event;
    const eventType = event.type;

    if (eventType === 'message_start') {
      this.messageStarted = true;
      this.contentBlocksStarted.clear();
      
      yield this.formatEvent('message_start', {
        type: 'message_start',
        message: {
          id: this.requestId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: event.message?.usage?.input_tokens || 0,
            output_tokens: 0,
          },
        },
      });
    } else if (eventType === 'content_block_start') {
      const index = event.index || 0;
      if (!this.contentBlocksStarted.has(index)) {
        this.contentBlocksStarted.add(index);
        yield this.formatEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: event.content_block,
        });
      }
    } else if (eventType === 'content_block_delta') {
      yield this.formatEvent('content_block_delta', {
        type: 'content_block_delta',
        index: event.index || 0,
        delta: event.delta,
      });
    } else if (eventType === 'content_block_stop') {
      yield this.formatEvent('content_block_stop', {
        type: 'content_block_stop',
        index: event.index || 0,
      });
    } else if (eventType === 'message_delta') {
      if (event.usage?.output_tokens) {
        this.totalOutputTokens = event.usage.output_tokens;
      }
      yield this.formatEvent('message_delta', {
        type: 'message_delta',
        delta: event.delta,
        usage: event.usage || undefined,
      });
    } else if (eventType === 'message_stop') {
      this.messageStarted = false;
      yield this.formatEvent('message_stop', {
        type: 'message_stop',
      });
    }
  }

  /**
   * 格式化 SSE 事件
   */
  private formatEvent(eventType: string, data: any): string {
    return `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
  }

  getMetrics() {
    return {
      totalInputTokens: this.totalInputTokens,
      totalOutputTokens: this.totalOutputTokens,
    };
  }
}

// ============================================================================
// SDK 选项构建器
// ============================================================================

function buildSDKOptions(request: MessagesRequest): Options {
  const options: Options = {
    pathToClaudeCodeExecutable: CONFIG.claude.cliPath,
    cwd: request.cwd || CONFIG.claude.defaultCwd,
    allowedTools: CONFIG.claude.defaultAllowedTools,
    permissionMode: request.permission_mode || CONFIG.claude.permissionMode,
    maxTurns: request.max_turns || CONFIG.claude.maxTurns,
    model: request.model !== CONFIG.claude.defaultModel ? request.model : undefined,
    includePartialMessages: request.stream || false,
  };

  // System prompt
  let systemPrompt: string | undefined;
  if (request.messages[0]?.role === 'system') {
    const systemMsg = request.messages[0];
    systemPrompt = typeof systemMsg.content === 'string' 
      ? systemMsg.content 
      : systemMsg.content.filter(b => b.type === 'text').map(b => (b as any).text).join('\n');
  } else if (request.system) {
    systemPrompt = typeof request.system === 'string' 
      ? request.system 
      : request.system.filter((b: any) => b.type === 'text').map((b: any) => b.text).join('\n');
  }

  if (systemPrompt) {
    options.systemPrompt = systemPrompt;
  }

  // Thinking tokens
  if (request.max_thinking_tokens !== undefined) {
    options.maxThinkingTokens = request.max_thinking_tokens;
  } else if (CONFIG.thinking.enabled) {
    options.maxThinkingTokens = CONFIG.thinking.defaultMaxTokens;
  }

  // 环境变量
  options.env = {};
  if (options.maxThinkingTokens) {
    options.env.MAX_THINKING_TOKENS = String(options.maxThinkingTokens);
  }
  if (request.temperature !== undefined) {
    options.env.ANTHROPIC_TEMPERATURE = String(request.temperature);
  }
  if (request.top_p !== undefined) {
    options.env.ANTHROPIC_TOP_P = String(request.top_p);
  }
  if (request.top_k !== undefined) {
    options.env.ANTHROPIC_TOP_K = String(request.top_k);
  }

  return options;
}

// ============================================================================
// 日志工具
// ============================================================================

function log(level: 'info' | 'error' | 'debug', message: string, data?: any) {
  if (level === 'debug' && !CONFIG.debug) return;
  
  const timestamp = new Date().toISOString();
  const prefix = {
    info: '✅',
    error: '❌',
    debug: '🔍',
  }[level];

  console.log(`[${timestamp}] ${prefix} ${message}`);
  if (data) {
    console.log(JSON.stringify(data, null, 2));
  }
}

// ============================================================================
// HTTP 服务器
// ============================================================================

const app = Fastify({
  logger: CONFIG.debug,
  requestIdLogLabel: 'request_id',
});

// CORS
app.register(cors, {
  origin: true,
  credentials: true,
});

// 根路由
app.get('/', async () => {
  return {
    service: 'Claude Messages API Gateway',
    version: '2.0.0',
    status: 'running',
    features: {
      streaming: true,
      history_replay: true,
      thinking: CONFIG.thinking.enabled,
      tools: true,
      images: true,
    },
    statistics,
  };
});

// 健康检查
app.get('/health', async () => {
  return {
    status: 'healthy',
    timestamp: Date.now(),
  };
});

// Token 计数 (简单估算)
app.post('/v1/messages/count-tokens', async (request: FastifyRequest) => {
  const body = request.body as any;
  const messages = body.messages || [];
  const system = body.system || '';

  let totalChars = system.length;
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

  return {
    input_tokens: Math.floor(totalChars * 0.5),
  };
});

// Messages API 主端点
app.post('/v1/messages', async (request: FastifyRequest, reply: FastifyReply) => {
  statistics.totalRequests++;
  const requestId = `msg_${randomUUID().replace(/-/g, '').slice(0, 24)}`;

  try {
    // 验证请求
    const validatedRequest = MessagesRequestSchema.parse(request.body);

    log('info', `New request [${requestId}]`, {
      model: validatedRequest.model,
      messageCount: validatedRequest.messages.length,
      stream: validatedRequest.stream,
    });

    // 验证最后一条消息
    const messages = validatedRequest.messages.filter(m => m.role !== 'system');
    if (messages.length === 0 || messages[messages.length - 1].role !== 'user') {
      throw new Error('Last message must be a user message');
    }

    // 构建 SDK 选项
    const options = buildSDKOptions(validatedRequest);

    // 流式响应
    if (validatedRequest.stream) {
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Request-ID': requestId,
      });

      const converter = new StreamConverter(validatedRequest.model, requestId);

      try {
        // 创建历史重放生成器
        const historyGenerator = HistoryReplayConverter.convertToStreamingInput(messages, requestId);
        
        // 执行查询
        const queryResult = query({
          prompt: historyGenerator,
          options,
        });

        // 转换并发送流
        for await (const chunk of converter.convert(queryResult)) {
          reply.raw.write(chunk);
        }

        const metrics = converter.getMetrics();
        log('info', `Stream completed [${requestId}]`, metrics);

        statistics.successfulRequests++;
      } catch (error) {
        log('error', `Stream error [${requestId}]`, error);
        statistics.failedRequests++;
        
        reply.raw.write(
          `event: error\ndata: ${JSON.stringify({
            type: 'error',
            error: {
              type: 'api_error',
              message: error instanceof Error ? error.message : String(error),
            },
          })}\n\n`
        );
      } finally {
        reply.raw.end();
      }
    } 
    // 非流式响应
    else {
      const allContent: any[] = [];
      let resultMessage: SDKResultMessage | null = null;

      // 创建历史重放生成器
      const historyGenerator = HistoryReplayConverter.convertToStreamingInput(messages, requestId);
      
      // 执行查询
      const queryResult = query({
        prompt: historyGenerator,
        options,
      });

      // 收集所有消息
      for await (const message of queryResult) {
        if (message.type === 'assistant') {
          const asstMsg = message as SDKAssistantMessage;
          for (const block of asstMsg.message.content) {
            if (typeof block === 'object' && block.type === 'text') {
              allContent.push({ type: 'text', text: block.text });
            } else if (typeof block === 'object' && block.type === 'thinking') {
              allContent.push({ type: 'thinking', thinking: (block as any).thinking });
            } else if (typeof block === 'object' && block.type === 'tool_use') {
              allContent.push({
                type: 'tool_use',
                id: block.id,
                name: block.name,
                input: block.input,
              });
            }
          }
        } else if (message.type === 'result') {
          resultMessage = message as SDKResultMessage;
        }
      }

      if (!resultMessage) {
        throw new Error('No result message received');
      }

      // 构建响应
      const response = {
        id: requestId,
        type: 'message' as const,
        role: 'assistant' as const,
        content: allContent.length > 0 ? allContent : [{ type: 'text', text: '' }],
        model: validatedRequest.model,
        stop_reason: resultMessage.subtype === 'success' ? 'end_turn' : 'error',
        stop_sequence: null,
        usage: {
          input_tokens: resultMessage.usage?.input_tokens || 0,
          output_tokens: resultMessage.usage?.output_tokens || 0,
        },
      };

      log('info', `Non-stream completed [${requestId}]`, {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      });

      statistics.successfulRequests++;
      return response;
    }
  } catch (error) {
    statistics.failedRequests++;
    log('error', `Request failed [${requestId}]`, error);

    const errorMessage = error instanceof Error ? error.message : String(error);
    const statusCode = errorMessage.includes('validation') ? 400 : 500;

    return reply.status(statusCode).send({
      type: 'error',
      error: {
        type: statusCode === 400 ? 'invalid_request_error' : 'api_error',
        message: errorMessage,
      },
    });
  }
});

// ============================================================================
// 启动服务器
// ============================================================================

async function start() {
  try {
    console.log('\n' + '='.repeat(70));
    console.log('🚀 Claude Messages API Gateway Starting...');
    console.log('='.repeat(70));
    console.log(`📍 Claude CLI Path: ${CONFIG.claude.cliPath}`);
    console.log(`📁 Default CWD: ${CONFIG.claude.defaultCwd}`);
    console.log(`🔧 Default Tools: ${CONFIG.claude.defaultAllowedTools.join(', ')}`);
    console.log(`🤖 Default Model: ${CONFIG.claude.defaultModel}`);
    console.log(`🧠 Thinking Enabled: ${CONFIG.thinking.enabled}`);
    if (CONFIG.thinking.enabled) {
      console.log(`💭 Default Thinking Tokens: ${CONFIG.thinking.defaultMaxTokens}`);
    }
    console.log(`🛠 Debug Mode: ${CONFIG.debug}`);
    console.log('='.repeat(70));

    await app.listen({
      host: CONFIG.server.host,
      port: CONFIG.server.port,
    });

    console.log(`✅ Server running on http://${CONFIG.server.host}:${CONFIG.server.port}`);
    console.log('='.repeat(70) + '\n');
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// 优雅关闭
process.on('SIGINT', async () => {
  console.log('\n' + '='.repeat(70));
  console.log('👋 Shutting down...');
  console.log(`📊 Total Requests: ${statistics.totalRequests}`);
  console.log(`✅ Successful: ${statistics.successfulRequests}`);
  console.log(`❌ Failed: ${statistics.failedRequests}`);
  console.log('='.repeat(70) + '\n');
  await app.close();
  process.exit(0);
});

start();
