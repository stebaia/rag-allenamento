import { createFileRoute } from '@tanstack/react-router'
import { DocumentUpload } from '#/components/document-upload'
import { DocumentList } from '#/components/document-list'

export const Route = createFileRoute('/_authenticated/documents')({
  component: DocumentsPage,
})

function DocumentsPage() {
  return (
    <div className="flex flex-col gap-6 rise-in">
      <div>
        <h1 className="display-title text-2xl font-semibold">Documenti</h1>
        <p className="text-sm text-muted-foreground">
          Carica i tuoi PDF o file di testo: verranno indicizzati e usati
          dalla chat per rispondere alle tue domande.
        </p>
      </div>
      <DocumentUpload />
      <DocumentList />
    </div>
  )
}
