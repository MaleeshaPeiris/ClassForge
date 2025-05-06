"use client"

import { useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"

// Sample data - in a real app, this would come from a database
const generateStudents = () => {
  const students = []

  for (let i = 0; i < 100; i++) {
    students.push({
      id: `student-${i}`,
      name: `Student ${i}`,
      grade: Math.floor(Math.random() * 6) + 7, // Grades 7-12
      academicScore: Math.round(50 + Math.random() * 50),
      wellbeingScore: Math.round(50 + Math.random() * 50),
      specialNeeds: Math.random() > 0.9,
      friendships: Math.round(Math.random() * 10),
      conflicts: Math.round(Math.random() * 3),
    })
  }

  return students
}

const students = generateStudents()

export default function StudentsPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [filteredStudents, setFilteredStudents] = useState(students)

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const term = e.target.value
    setSearchTerm(term)

    if (term.trim() === "") {
      setFilteredStudents(students)
    } else {
      setFilteredStudents(
        students.filter(
          (student) =>
            student.name.toLowerCase().includes(term.toLowerCase()) ||
            student.id.toLowerCase().includes(term.toLowerCase()) ||
            student.grade.toString().includes(term),
        ),
      )
    }
  }

  return (
    <DashboardShell>
      <DashboardHeader heading="Students" text="Manage student profiles and relationships.">
        <Button>Add Student</Button>
      </DashboardHeader>

      <Card>
        <CardHeader>
          <CardTitle>Student Directory</CardTitle>
          <CardDescription>View and manage all students in the system.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center mb-4">
            <Input
              placeholder="Search students..."
              value={searchTerm}
              onChange={handleSearch}
              className="max-w-sm"
            />
          </div>

          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Student</TableHead>
                  <TableHead>Grade</TableHead>
                  <TableHead>Academic Score</TableHead>
                  <TableHead>Wellbeing Score</TableHead>
                  <TableHead>Special Needs</TableHead>
                  <TableHead>Relationships</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredStudents.slice(0, 10).map((student) => (
                  <TableRow key={student.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src={`/placeholder.svg?height=32&width=32&query=student ${student.id}`} />
                          <AvatarFallback>{student.name.substring(0, 2)}</AvatarFallback>
                        </Avatar>
                        <div>
                          <div className="font-medium">{student.name}</div>
                          <div className="text-sm text-muted-foreground">{student.id}</div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>{student.grade}</TableCell>
                    <TableCell>{student.academicScore}</TableCell>
                    <TableCell>{student.wellbeingScore}</TableCell>
                    <TableCell>
                      {student.specialNeeds ? (
                        <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                          Yes
                        </Badge>
                      ) : (
                        "No"
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex space-x-1">
                        <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                          {student.friendships} Friends
                        </Badge>
                        {student.conflicts > 0 && (
                          <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                            {student.conflicts} Conflicts
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex space-x-2">
                        <Button variant="ghost" size="sm">
                          Edit
                        </Button>
                        <Button variant="ghost" size="sm">
                          View
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </DashboardShell>
  )
}