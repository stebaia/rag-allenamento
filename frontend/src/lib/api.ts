const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

const TOKEN_KEY = 'rag_token'

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return window.localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  window.localStorage.removeItem(TOKEN_KEY)
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken()
  const headers = new Headers(options.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const res = await fetch(`${API_URL}${path}`, { ...options, headers })

  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      message = body.detail ?? message
    } catch {
      // risposta senza corpo JSON, teniamo statusText
    }
    throw new ApiError(res.status, message)
  }

  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export interface TokenOut {
  access_token: string
  token_type: string
}

export function register(email: string, password: string) {
  return request<TokenOut>('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export function login(email: string, password: string) {
  const form = new URLSearchParams()
  form.set('username', email)
  form.set('password', password)
  return request<TokenOut>('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form,
  })
}

export type DocumentStatus = 'processing' | 'ready' | 'error'

export interface DocumentOut {
  id: string
  nome_file: string
  stato: DocumentStatus
  errore: string | null
  n_chunk: number
}

export function listDocuments() {
  return request<Array<DocumentOut>>('/documents')
}

export function getDocument(id: string) {
  return request<DocumentOut>(`/documents/${id}`)
}

export function uploadDocument(file: File) {
  const form = new FormData()
  form.append('file', file)
  return request<DocumentOut>('/documents', {
    method: 'POST',
    body: form,
  })
}

export function deleteDocument(id: string) {
  return request<void>(`/documents/${id}`, { method: 'DELETE' })
}

export interface RispostaOut {
  risposta: string
}

export interface MessaggioStorico {
  ruolo: 'user' | 'assistant'
  contenuto: string
}

export function ask(domanda: string, storico: Array<MessaggioStorico> = []) {
  return request<RispostaOut>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ domanda, storico }),
  })
}
