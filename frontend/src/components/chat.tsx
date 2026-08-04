import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { Bot, Loader2, Plus, Send, Sparkles, User } from 'lucide-react'
import { toast } from 'sonner'
import { ApiError, ask, getConversation, resumeAsk } from '#/lib/api'
import type { AskOut } from '#/lib/api'
import { Button } from '#/components/ui/button'
import { MarkdownLite } from '#/components/markdown-lite'
import { Textarea } from '#/components/ui/textarea'
import { cn } from '#/lib/utils'
import { useChatMessages } from '#/lib/chat-context'
import type { Message } from '#/lib/chat-context'

export function Chat() {
  const {
    messages,
    setMessages,
    conversationId,
    setConversationId,
    resetConversation,
  } = useChatMessages()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // L'id sopravvive al refresh in localStorage, i messaggi no: li ripeschiamo
  // dal checkpointer. Fissiamo l'id del mount così la query non riparte a ogni
  // risposta (dopo la prima domanda conversationId cambia da null all'id vero).
  const [idDaRipristinare] = useState(conversationId)

  const ripristino = useQuery({
    queryKey: ['conversation', idDaRipristinare],
    queryFn: () => getConversation(idDaRipristinare!),
    enabled: idDaRipristinare !== null,
    staleTime: Infinity,
    retry: false,
  })

  useEffect(() => {
    if (!ripristino.data) return
    setMessages(
      ripristino.data.messaggi.map((m) => ({
        id: crypto.randomUUID(),
        role: m.ruolo,
        content: m.contenuto,
      })),
    )
  }, [ripristino.data, setMessages])

  // Se l'id salvato non è più valido (checkpoint cancellato, altro utente),
  // ripartiamo puliti invece di continuare a mandare un id morto.
  useEffect(() => {
    if (ripristino.isError) resetConversation()
  }, [ripristino.isError, resetConversation])

  // Modalità agente (Fase 2): l'LLM decide se e quante volte cercare, e può
  // salvare memorie a lungo termine. Con la conferma attiva, il grafo si
  // sospende prima di scrivere e aspetta l'ok dell'utente (Fase 4).
  const [agente, setAgente] = useState(false)
  const [confermaMemorie, setConfermaMemorie] = useState(true)
  const [attesaConferma, setAttesaConferma] = useState<string | null>(null)

  function mostraRisposta(data: AskOut) {
    setConversationId(data.conversation_id)
    // in_attesa: il grafo si è fermato dentro il tool `ricorda`. Non c'è una
    // risposta da mostrare, c'è una domanda da fare all'utente.
    if (data.in_attesa?.conferma_memoria) {
      setAttesaConferma(data.in_attesa.conferma_memoria)
      return
    }
    setAttesaConferma(null)
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: 'assistant', content: data.risposta },
    ])
  }

  function segnalaErrore(err: unknown) {
    const message =
      err instanceof ApiError ? err.message : 'Errore durante la richiesta'
    toast.error(message)
  }

  const mutation = useMutation({
    mutationFn: ({
      domanda,
      conversationId: cid,
    }: {
      domanda: string
      conversationId: string | null
    }) => ask(domanda, cid, agente, confermaMemorie),
    // Alla prima risposta il backend ci assegna l'id della conversazione:
    // da qui in poi lo rimandiamo a ogni domanda.
    onSuccess: mostraRisposta,
    onError: (err) => {
      segnalaErrore(err)
      setMessages((prev) => prev.slice(0, -1))
    },
  })

  const ripresa = useMutation({
    mutationFn: (risposta: string) => resumeAsk(conversationId!, risposta),
    onSuccess: mostraRisposta,
    onError: (err) => {
      segnalaErrore(err)
      setAttesaConferma(null)
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, mutation.isPending])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const domanda = input.trim()
    // Con una conferma pendente il grafo è sospeso: una nuova domanda sullo
    // stesso thread verrebbe interpretata come risposta all'interrupt.
    if (!domanda || mutation.isPending || attesaConferma) return

    // I messaggi restano nel client solo per disegnare la UI: lo storico vero
    // vive nel checkpointer, indicizzato da conversationId.
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content: domanda },
    ])
    setInput('')
    mutation.mutate({ domanda, conversationId })
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex h-[calc(100vh-8.5rem)] flex-col rounded-xl border bg-card">
      <div className="flex items-center justify-between border-b px-4 py-2">
        <span className="text-sm font-medium text-muted-foreground">
          {conversationId ? 'Conversazione in corso' : 'Nuova conversazione'}
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant={agente ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setAgente((v) => !v)}
            disabled={mutation.isPending || ripresa.isPending}
            title={
              agente
                ? "Modalità agente: l'LLM decide se e quante volte cercare, e può salvare memorie"
                : 'Modalità deterministica: pipeline fissa recupera → genera'
            }
          >
            <Bot className="mr-1 size-4" />
            Agente
          </Button>
          {agente && (
            <Button
              variant={confermaMemorie ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => setConfermaMemorie((v) => !v)}
              disabled={mutation.isPending || ripresa.isPending}
              title="Chiedi conferma prima di salvare una memoria permanente"
            >
              Conferma memorie
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setAttesaConferma(null)
              resetConversation()
            }}
            disabled={messages.length === 0 || mutation.isPending}
          >
            <Plus className="mr-1 size-4" />
            Nuova chat
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-6">
        {/* isFetching, non isLoading: una query disabilitata resta in stato
            "pending" per sempre e nasconderebbe la chat vuota. */}
        {ripristino.isFetching ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        ) : messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="flex flex-col gap-4">
            {messages.map((msg) => (
              <ChatBubble key={msg.id} message={msg} />
            ))}
            {(mutation.isPending || ripresa.isPending) && <TypingBubble />}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Il grafo è sospeso dentro il tool `ricorda`: finché non rispondiamo,
          lo stato resta congelato nel checkpointer e nulla viene scritto. */}
      {attesaConferma && (
        <div className="border-t bg-muted/40 px-4 py-3">
          <p className="text-sm">
            Vuoi che ricordi questo?{' '}
            <span className="font-medium">«{attesaConferma}»</span>
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Le memorie valgono per tutte le conversazioni, non solo questa.
          </p>
          <div className="mt-2 flex gap-2">
            <Button
              size="sm"
              onClick={() => ripresa.mutate('sì')}
              disabled={ripresa.isPending}
            >
              Salva
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => ripresa.mutate('no')}
              disabled={ripresa.isPending}
            >
              Non salvare
            </Button>
          </div>
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-2 border-t p-3"
      >
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            attesaConferma
              ? 'Rispondi alla richiesta di conferma qui sopra...'
              : 'Chiedi qualcosa sui tuoi documenti...'
          }
          disabled={attesaConferma !== null}
          rows={1}
          className="max-h-40 min-h-10 flex-1 resize-none"
        />
        <Button
          type="submit"
          size="icon"
          disabled={!input.trim() || mutation.isPending || !!attesaConferma}
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
          'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed',
          isUser
            ? 'whitespace-pre-wrap rounded-tr-sm bg-primary text-primary-foreground'
            : 'rounded-tl-sm bg-muted text-foreground',
        )}
      >
        {/* Il markdown lo produce solo l'LLM: quello che scrive l'utente resta
            testo grezzo, con gli a capo preservati da whitespace-pre-wrap. */}
        {isUser ? message.content : <MarkdownLite testo={message.content} />}
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
