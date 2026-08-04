import { createFileRoute } from '@tanstack/react-router'
import { MemoryList } from '#/components/memory-list'

export const Route = createFileRoute('/_authenticated/memorie')({
  component: MemoriePage,
})

function MemoriePage() {
  return (
    <div className="flex flex-col gap-6 rise-in">
      <div>
        <h1 className="display-title text-2xl font-semibold">Memorie</h1>
        <p className="text-sm text-muted-foreground">
          Fatti che l'assistente ha imparato su di te e che valgono per{' '}
          <em>tutte</em> le conversazioni, non solo quella in corso. Non
          compaiono in nessuna chat: se uno è sbagliato, continua a influenzare
          le risposte finché non lo cancelli da qui.
        </p>
      </div>
      <MemoryList />
    </div>
  )
}
