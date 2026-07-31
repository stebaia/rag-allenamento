import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, CheckCircle2, Loader2, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { ApiError, deleteDocument, listDocuments } from '#/lib/api'
import type { DocumentOut } from '#/lib/api'
import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import { Skeleton } from '#/components/ui/skeleton'

const POLL_INTERVAL_MS = 2000

function StatusBadge({ doc }: { doc: DocumentOut }) {
  if (doc.stato === 'ready') {
    return (
      <Badge className="gap-1 bg-primary/15 text-primary hover:bg-primary/15">
        <CheckCircle2 className="size-3" />
        Pronto · {doc.n_chunk} chunk
      </Badge>
    )
  }
  if (doc.stato === 'error') {
    return (
      <Badge variant="destructive" className="gap-1">
        <AlertCircle className="size-3" />
        Errore
      </Badge>
    )
  }
  return (
    <Badge variant="secondary" className="gap-1">
      <Loader2 className="size-3 animate-spin" />
      Indicizzazione in corso
    </Badge>
  )
}

export function DocumentList() {
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    refetchInterval: (query) => {
      const docs = query.state.data
      const hasPending = docs?.some((d) => d.stato === 'processing')
      return hasPending ? POLL_INTERVAL_MS : false
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      toast.success('Documento eliminato')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (err) => {
      const message =
        err instanceof ApiError ? err.message : 'Eliminazione fallita'
      toast.error(message)
    },
  })

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-16 w-full rounded-lg" />
        ))}
      </div>
    )
  }

  if (!data || data.length === 0) {
    return (
      <p className="rounded-lg border border-dashed px-4 py-8 text-center text-sm text-muted-foreground">
        Nessun documento caricato. Carica un PDF o TXT per iniziare.
      </p>
    )
  }

  return (
    <ul className="flex flex-col gap-2">
      {data.map((doc) => (
        <li
          key={doc.id}
          className="flex items-center justify-between gap-3 rounded-lg border bg-card px-4 py-3"
        >
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{doc.nome_file}</p>
            {doc.stato === 'error' && doc.errore && (
              <p className="mt-0.5 truncate text-xs text-destructive">
                {doc.errore}
              </p>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <StatusBadge doc={doc} />
            <Button
              variant="ghost"
              size="icon"
              aria-label={`Elimina ${doc.nome_file}`}
              disabled={deleteMutation.isPending}
              onClick={() => deleteMutation.mutate(doc.id)}
            >
              <Trash2 className="size-4 text-muted-foreground" />
            </Button>
          </div>
        </li>
      ))}
    </ul>
  )
}
