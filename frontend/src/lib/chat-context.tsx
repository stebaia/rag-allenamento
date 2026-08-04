import { createContext, use, useCallback, useEffect, useState } from 'react'
import type { ReactNode } from 'react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

const CONVERSATION_KEY = 'rag_conversation_id'

interface ChatContextValue {
  messages: Array<Message>
  setMessages: React.Dispatch<React.SetStateAction<Array<Message>>>
  // Identifica la conversazione lato server: è la chiave con cui il
  // checkpointer di LangGraph ricarica lo storico. null = nuova conversazione.
  conversationId: string | null
  setConversationId: React.Dispatch<React.SetStateAction<string | null>>
  // Azzera chat e id: la prossima domanda aprirà un thread nuovo.
  resetConversation: () => void
}

const ChatContext = createContext<ChatContextValue | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<Array<Message>>([])
  // Lazy initializer: legge localStorage una sola volta al mount, non a ogni
  // render. Il guard su window serve se il bundle viene renderizzato server-side.
  const [conversationId, setConversationId] = useState<string | null>(() =>
    typeof window === 'undefined'
      ? null
      : window.localStorage.getItem(CONVERSATION_KEY),
  )

  // Tiene localStorage allineato allo state, incluso il caso null (nuova chat).
  useEffect(() => {
    if (conversationId) {
      window.localStorage.setItem(CONVERSATION_KEY, conversationId)
    } else {
      window.localStorage.removeItem(CONVERSATION_KEY)
    }
  }, [conversationId])

  // useCallback: identità stabile tra i render, così i componenti che la usano
  // come dipendenza di un effect non lo rieseguono a ogni render del provider.
  const resetConversation = useCallback(() => {
    setMessages([])
    setConversationId(null)
  }, [])

  return (
    <ChatContext
      value={{
        messages,
        setMessages,
        conversationId,
        setConversationId,
        resetConversation,
      }}
    >
      {children}
    </ChatContext>
  )
}

export function useChatMessages() {
  const ctx = use(ChatContext)
  if (!ctx) throw new Error('useChatMessages must be used within ChatProvider')
  return ctx
}
