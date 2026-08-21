// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 DEBUG LOGGING TOGGLE - Logs incoming model/system prompt + routing decisions to Render logs
const DEBUG_LOGGING = false; // Set to false once you've confirmed everything routes correctly

// 🔥 SYSTEM PROMPT FOLDING TOGGLE - Some open-weight models don't weight the
// `system` role as strongly as OpenAI/Anthropic models. If your Global/Custom
// Prompt seems to be getting ignored even though logs confirm it's arriving,
// try flipping this to true. It merges system message(s) into the first user
// turn instead of sending them as a separate `system` role.
const FOLD_SYSTEM_INTO_USER = false;

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'z-ai/glm-5.2',
  'gpt-4': 'minimaxai/minimax-m3',
  'gpt-4-turbo': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o': 'stepfun-ai/step-3.7-flash',
  'claude-3-opus': 'nvidia/nemotron-3-super-120b-a12b',
  'claude-3-sonnet': 'openai/gpt-oss-120b',
  'gemini-pro': 'deepseek-ai/deepseek-v4-flash-0731'
};

// Merge all `system` role messages into the first `user` message.
// Returns a new array; does not mutate the input.
function foldSystemIntoUser(messages) {
  const systemMessages = messages.filter(m => m.role === 'system');
  if (systemMessages.length === 0) return messages;

  const systemContent = systemMessages.map(m => m.content).join('\n\n');
  const rest = messages.filter(m => m.role !== 'system');

  const firstUserIdx = rest.findIndex(m => m.role === 'user');
  if (firstUserIdx === -1) {
    // No user turn yet (shouldn't normally happen) - inject one at the front
    return [{ role: 'user', content: systemContent }, ...rest];
  }

  const merged = [...rest];
  merged[firstUserIdx] = {
    ...merged[firstUserIdx],
    content: `${systemContent}\n\n${merged[firstUserIdx].content}`
  };
  return merged;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    debug_logging: DEBUG_LOGGING,
    fold_system_into_user: FOLD_SYSTEM_INTO_USER
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      top_p,
      frequency_penalty,
      presence_penalty,
      stream
    } = req.body;

    if (DEBUG_LOGGING) {
      const systemMsg = messages.find(m => m.role === 'system');
      console.log(`[REQUEST] incoming model="${model}" | messages=${messages.length} | system msg: ${
        systemMsg ? `"${systemMsg.content.slice(0, 120)}${systemMsg.content.length > 120 ? '...' : ''}"` : '(none found!)'
      }`);
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        const testRes = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        });

        if (testRes.status >= 200 && testRes.status < 300) {
          nimModel = model;
        } else if (DEBUG_LOGGING) {
          console.error(`[MODEL CHECK] "${model}" returned status ${testRes.status}:`, testRes.data);
        }
      } catch (e) {
        if (DEBUG_LOGGING) {
          console.error(`[MODEL CHECK FAILED] "${model}":`, e.response?.status, e.response?.data || e.message);
        }
      }

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
        if (DEBUG_LOGGING) {
          console.warn(`[FALLBACK] "${model}" did not match MODEL_MAPPING and failed live check -> falling back to "${nimModel}"`);
        }
      }
    }

    if (DEBUG_LOGGING) {
      console.log(`[ROUTING] "${model}" -> "${nimModel}"`);
    }

    const processedMessages = FOLD_SYSTEM_INTO_USER ? foldSystemIntoUser(messages) : messages;

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      top_p,
      frequency_penalty,
      presence_penalty,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = '';

                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      if (DEBUG_LOGGING) {
        console.log(`[RESPONSE] finish_reason=${response.data.choices?.[0]?.finish_reason} | usage=${JSON.stringify(response.data.usage || {})}`);
      }

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message, error.response?.data ? JSON.stringify(error.response.data) : '');

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Debug logging: ${DEBUG_LOGGING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Fold system into user: ${FOLD_SYSTEM_INTO_USER ? 'ENABLED' : 'DISABLED'}`);
});
