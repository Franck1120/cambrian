import type { Model } from '../types'

export const MODELS: Model[] = [
  // Groq
  { id: 'llama-3.3-70b', name: 'Llama 3.3 70B', provider: 'groq', context_window: 131072, speed: 'fast', cost_per_1k: 0.59, status: 'online' },
  { id: 'llama-3.1-8b', name: 'Llama 3.1 8B', provider: 'groq', context_window: 131072, speed: 'fast', cost_per_1k: 0.05, status: 'online' },
  { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', provider: 'groq', context_window: 32768, speed: 'fast', cost_per_1k: 0.27, status: 'online' },
  { id: 'gemma2-9b', name: 'Gemma 2 9B', provider: 'groq', context_window: 8192, speed: 'fast', cost_per_1k: 0.20, status: 'online' },
  { id: 'qwen-qwq-32b', name: 'QwQ 32B', provider: 'groq', context_window: 131072, speed: 'medium', cost_per_1k: 0.29, status: 'online' },
  { id: 'deepseek-r1-distill-70b', name: 'DeepSeek R1 Distill 70B', provider: 'groq', context_window: 131072, speed: 'medium', cost_per_1k: 0.75, status: 'rate-limited' },

  // Anthropic
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', provider: 'anthropic', context_window: 200000, speed: 'fast', cost_per_1k: 0.25, status: 'online' },
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5', provider: 'anthropic', context_window: 200000, speed: 'medium', cost_per_1k: 3.00, status: 'online' },
  { id: 'claude-opus-4-5', name: 'Claude Opus 4.5', provider: 'anthropic', context_window: 200000, speed: 'slow', cost_per_1k: 15.00, status: 'online' },

  // OpenAI
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openai', context_window: 128000, speed: 'fast', cost_per_1k: 0.15, status: 'online' },
  { id: 'gpt-4o', name: 'GPT-4o', provider: 'openai', context_window: 128000, speed: 'medium', cost_per_1k: 2.50, status: 'online' },
  { id: 'o1-mini', name: 'o1-mini', provider: 'openai', context_window: 128000, speed: 'slow', cost_per_1k: 3.00, status: 'rate-limited' },

  // Local
  { id: 'ollama-llama3', name: 'Llama 3 (Ollama)', provider: 'local', context_window: 4096, speed: 'slow', cost_per_1k: 0.00, status: 'online' },
  { id: 'ollama-qwen2.5', name: 'Qwen 2.5 7B (Ollama)', provider: 'local', context_window: 32768, speed: 'slow', cost_per_1k: 0.00, status: 'offline' },

  // Proxy
  { id: 'cli-proxy-api', name: 'CLI Proxy API', provider: 'proxy', context_window: 200000, speed: 'medium', cost_per_1k: 1.00, status: 'online' },
]

export const MODEL_PROVIDER_LABELS: Record<string, string> = {
  groq: 'Groq',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  local: 'Local',
  proxy: 'Proxy',
}

export const MODEL_PROVIDER_COLORS: Record<string, string> = {
  groq: '#00f0ff',
  openai: '#00ff88',
  anthropic: '#ff9500',
  local: '#a78bfa',
  proxy: '#fb923c',
}
