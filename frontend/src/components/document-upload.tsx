import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { Loader2, UploadCloud } from 'lucide-react'
import { toast } from 'sonner'
import { ApiError, uploadDocument } from '#/lib/api'
import { cn } from '#/lib/utils'

const VALID_EXTENSIONS = ['.pdf', '.txt']

function hasValidExtension(filename: string) {
  const lower = filename.toLowerCase()
  return VALID_EXTENSIONS.some((ext) => lower.endsWith(ext))
}

export function DocumentUpload() {
  const queryClient = useQueryClient()
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  const mutation = useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => {
      toast.success('Documento caricato, indicizzazione in corso')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (err) => {
      const message =
        err instanceof ApiError ? err.message : 'Caricamento fallito'
      toast.error(message)
    },
  })

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    const file = files[0]
    if (!hasValidExtension(file.name)) {
      toast.error('Sono supportati solo file PDF e TXT')
      return
    }
    mutation.mutate(file)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Carica un documento PDF o TXT"
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        'flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors',
        isDragging
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50 hover:bg-accent/30',
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt"
        className="sr-only"
        onChange={(e) => {
          handleFiles(e.target.files)
          e.target.value = ''
        }}
      />
      {mutation.isPending ? (
        <Loader2 className="size-8 animate-spin text-primary" />
      ) : (
        <UploadCloud className="size-8 text-muted-foreground" />
      )}
      <div>
        <p className="text-sm font-medium">
          Trascina un file qui o clicca per selezionarlo
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Formati supportati: PDF, TXT
        </p>
      </div>
    </div>
  )
}
