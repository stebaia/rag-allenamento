import { useMutation } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { Loader2, Send, Sparkles, User } from 'lucide-react'
import { toast } from 'sonner'
import { ApiError, ask } from '#/lib/api'
import { Button } from '#/components/ui/button'
import { Textarea } from '#/components/ui/textarea'
import { cn } from '#/lib/utils'
import { useChatMessages } from '#/lib/chat-context'
import type { Message } from '#/lib/chat-context'

export function Chat() {
  const { messages, setMessages } = useChatMessages()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const mutation = useMutation({
    mutationFn: ask,
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'assistant', content: data.risposta },
      ])
    },
    onError: (err) => {
      const message =
        err instanceof ApiError ? err.message : 'Errore durante la richiesta'
      toast.error(message)
      setMessages((prev) => prev.slice(0, -1))
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, mutation.isPending])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const domanda = input.trim()
    if (!domanda || mutation.isPending) return

    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content: domanda },
    ])
    setInput('')
    mutation.mutate(domanda)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex h-[calc(100vh-8.5rem)] flex-col rounded-xl border bg-card">
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="flex flex-col gap-4">
            {messages.map((msg) => (
              <ChatBubble key={msg.id} message={msg} />
            ))}
            {mutation.isPending && <TypingBubble />}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-2 border-t p-3"
      >
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Chiedi qualcosa sui tuoi documenti..."
          rows={1}
          className="max-h-40 min-h-10 flex-1 resize-none"
        />
        <Button
          type="submit"
          size="icon"
          disabled={!input.trim() || mutation.isPending}
          aria-label="Invia messaggio"
        >
          {mutation.isPending ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Send className="size-4" />
          )}
        </Button>
      </form>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-muted-foreground">
      <Sparkles className="size-8 text-primary" />
      <p className="max-w-xs text-sm">
        Fai una domanda sui documenti che hai caricato: dieta, allenamento,
        spesa, tutto quello che hai indicizzato.
      </p>
    </div>
  )
}

function ChatBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  return (
    <div
      className={cn('flex items-start gap-2', isUser && 'flex-row-reverse')}
    >
      <div
        className={cn(
          'flex size-7 shrink-0 items-center justify-center rounded-full',
          isUser ? 'bg-secondary' : 'bg-primary/15',
        )}
      >
        {isUser ? (
          <User className="size-3.5 text-secondary-foreground" />
        ) : (
          <Sparkles className="size-3.5 text-primary" />
        )}
      </div>
      <div
        className={cn(
          'max-w-[80%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
          isUser
            ? 'rounded-tr-sm bg-primary text-primary-foreground'
            : 'rounded-tl-sm bg-muted text-foreground',
        )}
      >
        {message.content}
      </div>
    </div>
  )
}

function TypingBubble() {
  return (
    <div className="flex items-start gap-2">
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/15">
        <Sparkles className="size-3.5 text-primary" />
      </div>
      <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm bg-muted px-4 py-3">
        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground" />
      </div>
    </div>
  )
}
