import { Fragment } from 'react'
import type { ReactNode } from 'react'

/**
 * Renderer markdown minimale per le risposte dell'LLM.
 *
 * Copre solo ciò che i modelli usano davvero qui: grassetto, corsivo, `code`
 * e liste puntate. Non è un parser completo — se un giorno servisse molto di
 * più, il rimpiazzo è `react-markdown`.
 *
 * Costruisce nodi React invece di HTML: niente dangerouslySetInnerHTML,
 * quindi il testo del modello non può iniettare markup nella pagina.
 */

// **grassetto**, *corsivo*, `codice`. Il gruppo esterno è catturante, così
// split() conserva i delimitatori e possiamo trasformarli a coppie.
const INLINE = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g

function inline(testo: string, chiave: string): Array<ReactNode> {
  return testo.split(INLINE).map((pezzo, i) => {
    const k = `${chiave}-${i}`
    if (pezzo.startsWith('**') && pezzo.endsWith('**') && pezzo.length > 4) {
      return <strong key={k}>{pezzo.slice(2, -2)}</strong>
    }
    if (pezzo.startsWith('`') && pezzo.endsWith('`') && pezzo.length > 2) {
      return (
        <code key={k} className="rounded bg-black/10 px-1 py-0.5 text-[0.9em]">
          {pezzo.slice(1, -1)}
        </code>
      )
    }
    if (pezzo.startsWith('*') && pezzo.endsWith('*') && pezzo.length > 2) {
      return <em key={k}>{pezzo.slice(1, -1)}</em>
    }
    return <Fragment key={k}>{pezzo}</Fragment>
  })
}

// "- voce", "* voce", "1. voce" — con eventuale rientro.
const VOCE_LISTA = /^\s*([-*]|\d+\.)\s+(.*)$/

export function MarkdownLite({ testo }: { testo: string }) {
  const righe = testo.split('\n')
  const blocchi: Array<ReactNode> = []
  let lista: Array<string> = []

  function chiudiLista() {
    if (lista.length === 0) return
    const voci = lista
    lista = []
    blocchi.push(
      <ul key={`ul-${blocchi.length}`} className="my-1 list-disc pl-5">
        {voci.map((v, i) => (
          <li key={i}>{inline(v, `li-${blocchi.length}-${i}`)}</li>
        ))}
      </ul>,
    )
  }

  righe.forEach((riga, i) => {
    const voce = riga.match(VOCE_LISTA)
    if (voce) {
      lista.push(voce[2])
      return
    }
    chiudiLista()
    if (riga.trim() === '') return // le righe vuote diventano spaziatura
    blocchi.push(
      <p key={`p-${i}`} className="my-1 first:mt-0 last:mb-0">
        {inline(riga, `p-${i}`)}
      </p>,
    )
  })
  chiudiLista()

  return <>{blocchi}</>
}
