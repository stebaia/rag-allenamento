import { Outlet, createFileRoute, redirect } from '@tanstack/react-router'
import { AppShell } from '#/components/app-shell'
import { getToken } from '#/lib/api'
import { ChatProvider } from '#/lib/chat-context'

export const Route = createFileRoute('/_authenticated')({
  beforeLoad: () => {
    if (!getToken()) {
      throw redirect({ to: '/login' })
    }
  },
  component: AuthenticatedLayout,
})

function AuthenticatedLayout() {
  return (
    <ChatProvider>
      <AppShell>
        <Outlet />
      </AppShell>
    </ChatProvider>
  )
}
