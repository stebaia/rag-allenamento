import { Link, useRouterState } from '@tanstack/react-router'
import { Brain, FileText, LogOut, MessageCircle, Sparkles } from 'lucide-react'
import type { ReactNode } from 'react'
import { Button } from '#/components/ui/button'
import { ThemeToggle } from '#/components/theme-toggle'
import { useAuth } from '#/lib/auth'
import { cn } from '#/lib/utils'

const NAV_ITEMS = [
  { to: '/chat', label: 'Chat', icon: MessageCircle },
  { to: '/documents', label: 'Documenti', icon: FileText },
  { to: '/memorie', label: 'Memorie', icon: Brain },
] as const

export function AppShell({ children }: { children: ReactNode }) {
  const { logout } = useAuth()
  const pathname = useRouterState({ select: (s) => s.location.pathname })

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <Sparkles className="size-5 text-primary" />
            <span className="display-title text-lg font-semibold tracking-tight">
              RAG Allenamento
            </span>
          </div>

          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                className={cn(
                  'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground',
                  pathname.startsWith(to)
                    ? 'bg-accent text-accent-foreground'
                    : 'text-muted-foreground',
                )}
              >
                <Icon className="size-4" />
                {label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-1">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              aria-label="Esci"
              onClick={logout}
            >
              <LogOut className="size-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">
        {children}
      </main>
    </div>
  )
}
