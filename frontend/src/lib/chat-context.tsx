import { createContext, use, useState } from 'react'
import type { ReactNode } from 'react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

interface ChatContextValue {
  messages: Array<Message>
  setMessages: React.Dispatch<React.SetStateAction<Array<Message>>>
}

const ChatContext = createContext<ChatContextValue | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<Array<Message>>([])
  return (
    <ChatContext value={{ messages, setMessages }}>{children}</ChatContext>
  )
}

export function useChatMessages() {
  const ctx = use(ChatContext)
  if (!ctx) throw new Error('useChatMessages must be used within ChatProvider')
  return ctx
}
