import { createFileRoute } from '@tanstack/react-router'
import { Chat } from '#/components/chat'

export const Route = createFileRoute('/_authenticated/chat')({
  component: ChatPage,
})

function ChatPage() {
  return (
    <div className="flex flex-col gap-4 rise-in">
      <div>
        <h1 className="display-title text-2xl font-semibold">Chat</h1>
        <p className="text-sm text-muted-foreground">
          Parla con l'assistente: risponde solo in base ai documenti che hai
          caricato.
        </p>
      </div>
      <Chat />
    </div>
  )
}
