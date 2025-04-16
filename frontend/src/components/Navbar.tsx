import Link from 'next/link'

export default function Navbar() {
  return (
    <nav className="bg-gray-900 text-white px-6 py-4 flex justify-between items-center">
      <h1 className="text-xl font-bold">ClassForge</h1>
      <div className="space-x-4">
        <Link href="/" className="hover:text-blue-300">Home</Link>
        <Link href="/students" className="hover:text-blue-300">Students</Link>
      </div>
    </nav>
  )
}