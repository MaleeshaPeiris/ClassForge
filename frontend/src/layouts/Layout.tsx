import { ReactNode } from 'react'
import Navbar from '@/components/Navbar'

type Props = {
  children: ReactNode
}

export default function Layout({ children }: Props) {
  return (
    <div>
      <Navbar />
      <main className="p-6">{children}</main>
    </div>
  )
}