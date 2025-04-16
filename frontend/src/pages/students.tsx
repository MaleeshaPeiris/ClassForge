import { useEffect, useState } from 'react'
import Layout from '@/layouts/Layout'

interface Student {
  id: number
  name: string
  grade: string
  wellbeing_score: number
}

export default function StudentsPage() {
  const [students, setStudents] = useState<Student[]>([])
  const [query, setQuery] = useState('')

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/students')
      .then(res => res.json())
      .then(data => setStudents(data))
      .catch(err => console.error('API error:', err))
  }, [])

  const filtered = students.filter(s =>
    s.name.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <Layout>
      <h1 className="text-2xl font-semibold mb-4">Student List</h1>
      <input
        type="text"
        placeholder="Search students..."
        value={query}
        onChange={e => setQuery(e.target.value)}
        className="border px-3 py-2 rounded w-full max-w-sm mb-6"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map(s => (
          <div key={s.id} className="border rounded p-4 shadow">
            <h2 className="text-lg font-bold">{s.name}</h2>
            <p>Grade: {s.grade}</p>
            <p>Wellbeing Score: {s.wellbeing_score}</p>
          </div>
        ))}
      </div>
    </Layout>
  )
}
