/**
 * ==================================================================================
 * haI-ts - Claude Messages API Gateway (Bun Single File Edition)
 * ==================================================================================
 * 
 * A TypeScript gateway that exposes Claude Code Agent as a standard 
 * OpenAI/Anthropic compatible REST API with history replay.
 * 
 * Runtime: Bun (optimized)
 * Framework: Hono
 * Agent: @anthropic-ai/claude-agent-sdk
 * 
 * Author: Max
 * Version: 2.0.0-bun
 */

 import { Hono } from 'hono';
 import { stream } from 'hono/streaming';
 import { logger } from 'hono/logger';
 import { cors } from 'hono/cors';
 import { 
   query, 
   type SDKMessage, 
   type Options as SDKOptions,
   type PermissionMode
 } from '@anthropic-ai/claude-agent-sdk';
 
 // ==================================================================================
 // CONFIGURATION
 // ==================================================================================
 
 const CONFIG = {
   API_HOST: process.env.API_HOST || '0.0.0.0',
   API_PORT: Number(process.env.API_PORT || 8000),
   DEBUG: process.env.DEBUG === 'true',
   
   // Agent Settings
   DEFAULT_CWD: process.env.DEFAULT_CWD || '/tmp',
   DEFAULT_MODEL: process.env.DEFAULT_MODEL || 'claude-3-5-sonnet-20241022',
   DEFAULT_MAX_THINKING_TOKENS: Number(process.env.DEFAULT_MAX_THINKING_TOKENS || 8000),
   ENABLE_THINKING: process.env.ENABLE_THINKING_BY_DEFAULT !== 'false',
   
   // Permission Mode
   PERMISSION_MODE: (process.env.DEFAULT_PERMISSION_MODE || 'acceptEdits') as PermissionMode,
   
   // Tools
   ALLOWED_TOOLS: (process.env.DEFAULT_ALLOWED_TOOLS || 'WebSearch,Bash,Read,Write,Edit,Glob,Grep').split(','),
   
   // Max turns
   MAX_TURNS: Number(process.env.DEFAULT_MAX_TURNS || 99999),
 };
 
 const ALL_STANDARD_TOOLS = [
   'Task',
   'Bash',
   'BashOutput',
   'Edit',
   'Read',
   'Write',
   'Glob',
   'Grep',
   'KillBash',
   'NotebookEdit',
   'WebFetch',
   'WebSearch',
   'TodoWrite',
   'ExitPlanMode',
   'ListMcpResources',
   'ReadMcpResource',
 ];
 
 
 // ==================================================================================
 // TYPE DEFINITIONS
 // ==================================================================================
 
 interface Message {
   role: 'user' | 'assistant' | 'system';
   content: string | ContentBlock[];
 }
 
 type ContentBlock = 
   | { type: 'text'; text: string }
   | { type: 'image'; source: ImageSource }
   | { type: 'tool_use'; id: string; name: string; input: any }
   | { type: 'tool_result'; tool_use_id: string; content: string | any[]; is_error?: boolean }
   | { type: 'thinking'; thinking: string };
 
 interface ImageSource {
   type: 'base64' | 'url';
   media_type?: string;
   data?: string;
   url?: string;
 }
 
 interface MessagesRequest {
   model: string;
   messages: Message[];
   system?: string;
   max_tokens?: number;
   stream?: boolean;
   temperature?: number;
   top_p?: number;
   top_k?: number;
   max_thinking_tokens?: number;
   cwd?: string;
   max_turns?: number;
 }
 
 interface Usage {
   input_tokens: number;
   output_tokens: number;
   cache_creation_input_tokens?: number;
   cache_read_input_tokens?: number;
 }
 
 interface MessagesResponse {
   id: string;
   type: 'message';
   role: 'assistant';
   content: ContentBlock[];
   model: string;
   stop_reason: string;
   usage: Usage;
 }
 
 // ==================================================================================
 // HISTORY REPLAY CONVERTER
 // ==================================================================================
 
 class HistoryReplayConverter {
   static convertMessagesToPrompt(messages: Message[]): string {
     if (!messages || messages.length === 0) {
       throw new Error('Messages array cannot be empty');
     }
 
     const history = messages.slice(0, -1);
     const current = messages[messages.length - 1];
 
     if (current.role !== 'user') {
       throw new Error('The last message must be from user role');
     }
 
     const parts: string[] = [];
 
     if (history.length > 0) {
       parts.push('<conversation_history>');
       parts.push('This is the previous conversation for context. You should be aware of it,');
       parts.push('but respond ONLY to the <current_question> below.');
       parts.push('');
 
       for (const msg of history) {
         if (msg.role === 'system') continue;
         
         const tag = msg.role === 'user' ? 'user' : 'assistant';
         parts.push(`<${tag}>`);
         parts.push(this.escapeXml(this.extractTextContent(msg)));
         parts.push(`</${tag}>`);
       }
 
       parts.push('</conversation_history>');
       parts.push('');
     }
 
     parts.push('<current_question>');
     parts.push(this.extractTextContent(current));
     parts.push('</current_question>');
 
     return parts.join('\n');
   }
 
   private static extractTextContent(msg: Message): string {
     if (typeof msg.content === 'string') {
       return msg.content;
     }
 
     return msg.content
       .map(block => {
         switch (block.type) {
           case 'text':
             return block.text;
           case 'tool_use':
             return `[Used tool: ${block.name} with id=${block.id}]`;
           case 'tool_result':
             const preview = typeof block.content === 'string' 
               ? block.content.substring(0, 200)
               : JSON.stringify(block.content).substring(0, 200);
             return `[Tool result for ${block.tool_use_id}]: ${preview}`;
           case 'image':
             return '[Image attached]';
           case 'thinking':
             return `[Thinking: ${block.thinking.substring(0, 100)}...]`;
           default:
             return '';
         }
       })
       .filter(Boolean)
       .join('\n');
   }
 
   private static escapeXml(text: string): string {
     return text
       .replace(/&/g, '&amp;')
       .replace(/</g, '&lt;')
       .replace(/>/g, '&gt;')
       .replace(/"/g, '&quot;')
       .replace(/'/g, '&apos;');
   }
 
 static formatSSE(event: any): string {
   return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
 }
 
   static convertSDKMessageToEvent(sdkMsg: SDKMessage): any {
     if (sdkMsg.type === 'stream_event') {
       return sdkMsg.event;
     }
 
     if (sdkMsg.type === 'result') {
       return {
         type: 'message_delta',
         delta: { stop_reason: sdkMsg.is_error ? 'error' : 'end_turn' },
         usage: sdkMsg.usage
       };
     }
 
     if (sdkMsg.type === 'assistant') {
       return {
         type: 'assistant_message',
         message: sdkMsg.message
       };
     }
 
     return null;
   }
 }
 
 // ==================================================================================
 // APPLICATION
 // ==================================================================================
 
 const app = new Hono();
 
 app.use('*', logger());
 app.use('*', cors());
 
 let requestCount = 0;
 let successCount = 0;
 let errorCount = 0;
 
 // ==================================================================================
 // ROUTES
 // ==================================================================================
 
 app.get('/', (c) => {
   return c.json({
     service: 'Claude Messages API Gateway (Bun)',
     version: '2.0.0-bun',
     status: 'running',
     runtime: 'bun',
     features: {
       streaming: true,
       history_replay: true,
       thinking: CONFIG.ENABLE_THINKING,
       tools: true,
     },
     statistics: {
       total_requests: requestCount,
       successful: successCount,
       failed: errorCount,
     }
   });
 });
 
 app.get('/health', (c) => {
   return c.json({ 
     status: 'healthy',
     timestamp: Date.now()
   });
 });
 
 app.post('/v1/messages', async (c) => {
   requestCount++;
   const requestId = `msg_${crypto.randomUUID().replace(/-/g, '').substring(0, 24)}`;
   
   // ==================================================================================
   // EXTRACT CUSTOM AUTH FROM HEADERS
   // ==================================================================================
   let customAuthToken: string | undefined;
   let customBaseUrl: string | undefined;
   
   const authHeader = c.req.header('authorization') || c.req.header('x-api-key') || '';
   
   if (authHeader && authHeader.includes('cc:')) {
     try {
       const ccIndex = authHeader.indexOf('cc:');
       const ccContent = authHeader.substring(ccIndex + 3);
       const parts = ccContent.split('!');
       
       if (parts.length >= 2 && parts[0].trim() && parts[1].trim()) {
         customAuthToken = parts[0].trim();
         customBaseUrl = parts[1].trim();
         
         if (CONFIG.DEBUG) {
           console.log(`🔑 Custom auth extracted:`);
           console.log(`   Token: ${customAuthToken.substring(0, 10)}...`);
           console.log(`   URL: ${customBaseUrl}`);
         }
       }
     } catch (e) {
       if (CONFIG.DEBUG) {
         console.warn(`⚠️  Failed to parse cc: header:`, e);
       }
     }
   }
   
   let body: MessagesRequest;
   
   try {
     body = await c.req.json();
   } catch (e) {
     errorCount++;
     return c.json({ 
       type: 'error',
       error: { message: 'Invalid JSON in request body' }
     }, 400);
   }
 
   if (!body.messages || body.messages.length === 0) {
     errorCount++;
     return c.json({
       type: 'invalid_request_error',
       error: { message: 'messages field is required and cannot be empty' }
     }, 400);
   }
 
   if (CONFIG.DEBUG) {
     console.log(`\n${'='.repeat(70)}`);
     console.log(`🔨 New Request [${requestId}]`);
     console.log(`   Model: ${body.model || CONFIG.DEFAULT_MODEL}`);
     console.log(`   Messages: ${body.messages.length}`);
     console.log(`   Streaming: ${body.stream ?? false}`);
     console.log(`${'='.repeat(70)}\n`);
   }
 
   let promptString: string;
   try {
     promptString = HistoryReplayConverter.convertMessagesToPrompt(body.messages);
   } catch (e: any) {
     errorCount++;
     return c.json({
       type: 'invalid_request_error',
       error: { message: e.message }
     }, 400);
   }
 
   let systemPrompt = body.system;
   if (!systemPrompt && body.messages[0]?.role === 'system') {
     systemPrompt = typeof body.messages[0].content === 'string' 
       ? body.messages[0].content 
       : body.messages[0].content.filter(b => b.type === 'text').map(b => (b as any).text).join('\n');
   }
 
   const sdkOptions: SDKOptions = {
     cwd: body.cwd || CONFIG.DEFAULT_CWD,
     model: body.model || CONFIG.DEFAULT_MODEL,
     permissionMode: CONFIG.PERMISSION_MODE,
     // Disable all tools
     allowedTools: [],
     disallowed_tools: ALL_STANDARD_TOOLS,
     maxTurns: body.max_turns || CONFIG.MAX_TURNS,
     includePartialMessages: true,
     systemPrompt: systemPrompt,
     settingSources: ['local'],
     env : {
       'ANTHROPIC_AUTH_TOKEN': 'xxxx',
       'ANTHROPIC_BASE_URL': 'https://www.88code.org/api',
     },
   };
 
   if (body.max_thinking_tokens !== undefined) {
     sdkOptions.maxThinkingTokens = body.max_thinking_tokens;
   } else if (CONFIG.ENABLE_THINKING) {
     sdkOptions.maxThinkingTokens = CONFIG.DEFAULT_MAX_THINKING_TOKENS;
   }
 
   if (CONFIG.DEBUG) {
     sdkOptions.stderr = (data: string) => console.error(`[Agent Stderr]: ${data}`);
   }
 
   // ================================================================================
   // STREAMING RESPONSE
   // ================================================================================
   if (body.stream) {
     return stream(c, async (stream) => {
       try {
         await stream.write(HistoryReplayConverter.formatSSE({
           type: 'message_start',
           message: {
             id: requestId,
             type: 'message',
             role: 'assistant',
             content: [],
             model: sdkOptions.model,
             stop_reason: null,
             usage: { input_tokens: 0, output_tokens: 0 }
           }
         }));
 
         const generator = query({
           prompt: promptString,
           options: sdkOptions
         });
 
         let eventCount = 0;
 
         for await (const sdkMsg of generator) {
           const event = HistoryReplayConverter.convertSDKMessageToEvent(sdkMsg);
           if (event && event.type) {
             await stream.write(HistoryReplayConverter.formatSSE(event));
             eventCount++;
           }
         }
 
         await stream.write(HistoryReplayConverter.formatSSE({
           type: 'message_stop'
         }));
 
         if (CONFIG.DEBUG) {
           console.log(`✅ Streaming complete [${requestId}]`);
           console.log(`   Events sent: ${eventCount}\n`);
         }
 
         successCount++;
 
       } catch (error: any) {
         errorCount++;
         console.error(`❌ Streaming error [${requestId}]:`, error);
         
         await stream.write(HistoryReplayConverter.formatSSE({
           type: 'error',
           error: {
             type: 'api_error',
             message: error.message
           }
         }));
       }
     });
   }
 
   // ================================================================================
   // NON-STREAMING RESPONSE
   // ================================================================================
   else {
     try {
       const generator = query({
         prompt: promptString,
         options: sdkOptions
       });
 
       const contentBlocks: ContentBlock[] = [];
       let usage: Usage = { input_tokens: 0, output_tokens: 0 };
       let stopReason = 'end_turn';
 
       for await (const sdkMsg of generator) {
         if (sdkMsg.type === 'stream_event') {
           const event = sdkMsg.event;
           
           if (event.type === 'content_block_start') {
             const block = event.content_block;
             if (block.type === 'text') {
               contentBlocks.push({ type: 'text', text: '' });
             } else if (block.type === 'thinking') {
               contentBlocks.push({ type: 'thinking', thinking: '' });
             } else if (block.type === 'tool_use') {
               contentBlocks.push({
                 type: 'tool_use',
                 id: block.id,
                 name: block.name,
                 input: block.input
               });
             }
           }
           
           if (event.type === 'content_block_delta') {
             const delta = event.delta;
             const index = event.index;
             
             if (delta.type === 'text_delta' && contentBlocks[index]?.type === 'text') {
               (contentBlocks[index] as any).text += delta.text;
             } else if (delta.type === 'thinking_delta' && contentBlocks[index]?.type === 'thinking') {
               (contentBlocks[index] as any).thinking += delta.thinking;
             }
           }
         }
         
         if (sdkMsg.type === 'result') {
           if (sdkMsg.usage) {
             usage = {
               input_tokens: sdkMsg.usage.input_tokens || 0,
               output_tokens: sdkMsg.usage.output_tokens || 0,
               cache_creation_input_tokens: sdkMsg.usage.cache_creation_input_tokens,
               cache_read_input_tokens: sdkMsg.usage.cache_read_input_tokens,
             };
           }
           
           if (sdkMsg.is_error) {
             stopReason = 'error';
           }
         }
       }
 
       const response: MessagesResponse = {
         id: requestId,
         type: 'message',
         role: 'assistant',
         content: contentBlocks.length > 0 ? contentBlocks : [{ type: 'text', text: '' }],
         model: sdkOptions.model || CONFIG.DEFAULT_MODEL,
         stop_reason: stopReason,
         usage: usage
       };
 
       if (CONFIG.DEBUG) {
         console.log(`✅ Non-streaming complete [${requestId}]`);
         console.log(`   Input tokens: ${usage.input_tokens}`);
         console.log(`   Output tokens: ${usage.output_tokens}\n`);
       }
 
       successCount++;
       return c.json(response);
 
     } catch (error: any) {
       errorCount++;
       console.error(`❌ Error [${requestId}]:`, error);
       
       return c.json({
         type: 'api_error',
         error: { message: error.message }
       }, 500);
     }
   }
 });
 
 app.post('/v1/messages/count-tokens', async (c) => {
   try {
     const body = await c.req.json();
     const messages = body.messages || [];
     const system = body.system || '';
     
     let totalChars = system.length;
     for (const msg of messages) {
       if (typeof msg.content === 'string') {
         totalChars += msg.content.length;
       } else if (Array.isArray(msg.content)) {
         for (const block of msg.content) {
           if (block.type === 'text') {
             totalChars += block.text.length;
           }
         }
       }
     }
     
     const estimatedTokens = Math.ceil(totalChars * 0.5);
     
     return c.json({ input_tokens: estimatedTokens });
   } catch (e) {
     return c.json({ error: 'Invalid request' }, 400);
   }
 });
 
 // ==================================================================================
 // SERVER STARTUP (Bun Native)
 // ==================================================================================
 
 console.log('\n' + '='.repeat(70));
 console.log('🚀 Claude Messages API Gateway (Bun) Starting...');
 console.log('='.repeat(70));
 console.log(`📍 CWD: ${CONFIG.DEFAULT_CWD}`);
 console.log(`🔧 Tools: ${CONFIG.ALLOWED_TOOLS.join(', ')}`);
 console.log(`🤖 Model: ${CONFIG.DEFAULT_MODEL}`);
 console.log(`🧠 Thinking: ${CONFIG.ENABLE_THINKING} (${CONFIG.DEFAULT_MAX_THINKING_TOKENS} tokens)`);
 console.log(`🔒 Permission Mode: ${CONFIG.PERMISSION_MODE}`);
 console.log(`🐛 Debug: ${CONFIG.DEBUG}`);
 console.log('='.repeat(70));
 console.log(`✅ Server running on http://${CONFIG.API_HOST}:${CONFIG.API_PORT}\n`);
 
 // 使用 Bun 原生 serve API
 export default {
   port: CONFIG.API_PORT,
   hostname: CONFIG.API_HOST,
   fetch: app.fetch,
 };
