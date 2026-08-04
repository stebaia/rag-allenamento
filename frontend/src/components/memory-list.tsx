import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Brain, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { ApiError, deleteMemory, listMemories } from '#/lib/api'
import { Button } from '#/components/ui/button'
import { Skeleton } from '#/components/ui/skeleton'

export function MemoryList() {
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['memories'],
    queryFn: listMemories,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteMemory,
    onSuccess: () => {
      toast.success('Memoria dimenticata')
      queryClient.invalidateQueries({ queryKey: ['memories'] })
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
        {[0, 1].map((i) => (
          <Skeleton key={i} className="h-14 w-full rounded-lg" />
        ))}
      </div>
    )
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed py-10 text-center text-muted-foreground">
        <Brain className="size-7" />
        <p className="max-w-sm text-sm">
          Nessuna memoria salvata. In modalità agente puoi dire "ricorda
          che..." e l'assistente conserverà il fatto per tutte le conversazioni.
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {data.map((m) => (
        <div
          key={m.id}
          className="flex items-center justify-between gap-3 rounded-lg border px-4 py-3"
        >
          <div className="flex min-w-0 items-center gap-3">
            <Brain className="size-4 shrink-0 text-primary" />
            <span className="truncate text-sm">{m.fatto}</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Dimentica: ${m.fatto}`}
            onClick={() => deleteMutation.mutate(m.id)}
            disabled={deleteMutation.isPending}
          >
            <Trash2 className="size-4 text-destructive" />
          </Button>
        </div>
      ))}
    </div>
  )
}
