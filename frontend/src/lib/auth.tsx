import {
  createContext,
  use,
  useCallback,
  useMemo,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import * as api from './api'

interface AuthContextValue {
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => api.getToken() !== null,
  )

  const login = useCallback(async (email: string, password: string) => {
    const { access_token } = await api.login(email, password)
    api.setToken(access_token)
    setIsAuthenticated(true)
  }, [])

  const register = useCallback(async (email: string, password: string) => {
    const { access_token } = await api.register(email, password)
    api.setToken(access_token)
    setIsAuthenticated(true)
  }, [])

  const logout = useCallback(() => {
    api.clearToken()
    setIsAuthenticated(false)
  }, [])

  const value = useMemo(
    () => ({ isAuthenticated, login, register, logout }),
    [isAuthenticated, login, register, logout],
  )

  return <AuthContext value={value}>{children}</AuthContext>
}

export function useAuth() {
  const ctx = use(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
