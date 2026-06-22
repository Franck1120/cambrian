import { useEffect, useRef, useState, useCallback } from 'react'
import type { WSMessage } from '../types'

export type WSStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface UseWebSocketOptions {
  url: string
  enabled?: boolean
  reconnectDelay?: number
  onMessage?: (msg: WSMessage) => void
}

interface UseWebSocketReturn {
  status: WSStatus
  lastMessage: WSMessage | null
  send: (msg: unknown) => void
  disconnect: () => void
}

export const useWebSocket = ({
  url,
  enabled = true,
  reconnectDelay = 3000,
  onMessage,
}: UseWebSocketOptions): UseWebSocketReturn => {
  const [status, setStatus] = useState<WSStatus>('disconnected')
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)
  // Store latest options in refs to avoid stale closure issues
  const enabledRef = useRef(enabled)
  const reconnectDelayRef = useRef(reconnectDelay)
  const onMessageRef = useRef(onMessage)

  useEffect(() => { enabledRef.current = enabled }, [enabled])
  useEffect(() => { reconnectDelayRef.current = reconnectDelay }, [reconnectDelay])
  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])

  const scheduleReconnect = useCallback(() => {
    if (!enabledRef.current || !mountedRef.current) return
    reconnectTimer.current = setTimeout(() => {
      if (!mountedRef.current) return
      attemptConnect()
    }, reconnectDelayRef.current)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const attemptConnect = useCallback(() => {
    if (!enabledRef.current || !mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        setStatus('connected')
      }

      ws.onmessage = (event: MessageEvent<string>) => {
        if (!mountedRef.current) return
        try {
          const msg = JSON.parse(event.data) as WSMessage
          setLastMessage(msg)
          onMessageRef.current?.(msg)
        } catch {
          // ignore malformed messages
        }
      }

      ws.onerror = () => {
        if (!mountedRef.current) return
        setStatus('error')
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setStatus('disconnected')
        scheduleReconnect()
      }
    } catch {
      setStatus('error')
    }
  }, [url, scheduleReconnect])

  useEffect(() => {
    mountedRef.current = true
    if (enabled) attemptConnect()
    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [enabled, attemptConnect])

  const send = useCallback((msg: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg))
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    wsRef.current?.close()
    setStatus('disconnected')
  }, [])

  return { status, lastMessage, send, disconnect }
}
