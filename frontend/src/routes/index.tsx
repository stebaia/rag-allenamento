import { createFileRoute, redirect } from '@tanstack/react-router'
import { getToken } from '#/lib/api'

export const Route = createFileRoute('/')({
  beforeLoad: () => {
    throw redirect({ to: getToken() ? '/chat' : '/login' })
  },
})
