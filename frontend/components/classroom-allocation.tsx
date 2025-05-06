"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

// Sample data - in a real app, this would come from the allocation algorithm
const generateClassrooms = () => {
  const classrooms = []

  for (let i = 0; i < 4; i++) {
    const students = []

    for (let j = 0; j < 25; j++) {
      students.push({
        id: `student-${i}-${j}`,
        name: `Student ${i * 25 + j}`,
        academicScore: Math.round(50 + Math.random() * 50),
        wellbeingScore: Math.round(50 + Math.random() * 50),
        friendships: Math.round(Math.random() * 5),
        conflicts: Math.round(Math.random() * 2),
      })
    }

    classrooms.push({
      id: `class-${i}`,
      name: `Class ${i + 1}`,
      students,
    })
  }

  return classrooms
}

const classrooms = generateClassrooms()

export function ClassroomAllocation() {
  const [classroomData, setClassroomData] = useState(classrooms)

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-medium">Classroom Allocation Results</h3>
          <p className="text-sm text-muted-foreground">Review the generated classroom allocations</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm">
            Reset
          </Button>
          <Button size="sm">Save Allocation</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {classroomData.map((classroom) => (
          <Card key={classroom.id} className="h-[400px]">
            <CardHeader className="pb-2">
              <CardTitle>{classroom.name}</CardTitle>
              <CardDescription>{classroom.students.length} students</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px] pr-4">
                {classroom.students.map((student) => (
                  <div
                    key={student.id}
                    className="flex items-center p-2 border rounded-md mb-2 bg-background"
                  >
                    <Avatar className="h-8 w-8 mr-2">
                      <AvatarImage src={`/placeholder.svg?height=32&width=32&query=student ${student.id}`} />
                      <AvatarFallback>{student.name.substring(0, 2)}</AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <div className="font-medium">{student.name}</div>
                      <div className="text-xs text-muted-foreground">
                        Academic: {student.academicScore} | Wellbeing: {student.wellbeingScore}
                      </div>
                    </div>
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
                  </div>
                ))}
              </ScrollArea>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Allocation Metrics</CardTitle>
          <CardDescription>Impact of current allocation on key metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-sm font-medium">Academic Balance</p>
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                  <div className="bg-green-600 h-2.5 rounded-full" style={{ width: "85%" }}></div>
                </div>
                <span className="text-sm font-medium">85%</span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-sm font-medium">Wellbeing Distribution</p>
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: "78%" }}></div>
                </div>
                <span className="text-sm font-medium">78%</span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-sm font-medium">Friendship Retention</p>
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                  <div className="bg-purple-600 h-2.5 rounded-full" style={{ width: "82%" }}></div>
                </div>
                <span className="text-sm font-medium">82%</span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-sm font-medium">Conflict Reduction</p>
              <div className="flex items-center">
                <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                  <div className="bg-red-600 h-2.5 rounded-full" style={{ width: "90%" }}></div>
                </div>
                <span className="text-sm font-medium">90%</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}