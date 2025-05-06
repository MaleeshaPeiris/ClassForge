"use client"

import { useState } from "react"
import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AllocationParameters } from "@/components/allocation-parameters"
import { NetworkVisualization } from "@/components/network-visualization"
import { ClassroomAllocation } from "@/components/classroom-allocation"
import { Progress } from "@/components/ui/progress"
import { toast } from "sonner"

export default function AllocatePage() {
  const [allocating, setAllocating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [allocated, setAllocated] = useState(false)


  const handleAllocate = () => {
    setAllocating(true)
    setProgress(0)

    // Simulate allocation process
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setAllocating(false)
          setAllocated(true)
          toast({
            title: "Allocation Complete",
            description: "Classroom allocation has been successfully generated.",
          })
          return 100
        }
        return prev + 5
      })
    }, 200)
  }

  return (
    <DashboardShell>
      <DashboardHeader
        heading="Classroom Allocation"
        text="Configure and generate optimal classroom allocations based on social network analysis."
      >
        <div className="flex space-x-2">
          <Button variant="outline">Save Configuration</Button>
          <Button onClick={handleAllocate} disabled={allocating}>
            {allocating ? "Allocating..." : "Generate Allocation"}
          </Button>
        </div>
      </DashboardHeader>

      <Tabs defaultValue="parameters" className="space-y-4">
        <TabsList>
          <TabsTrigger value="parameters">Parameters</TabsTrigger>
          <TabsTrigger value="network">Network Analysis</TabsTrigger>
          <TabsTrigger value="results" disabled={!allocated}>
            Results
          </TabsTrigger>
        </TabsList>

        <TabsContent value="parameters" className="space-y-4">
          <AllocationParameters />

          {allocating && (
            <Card>
              <CardHeader>
                <CardTitle>Allocation in Progress</CardTitle>
                <CardDescription>
                  Our AI is analyzing social networks and optimizing classroom assignments.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Progress value={progress} className="h-2 mb-2" />
                <div className="text-sm text-muted-foreground">
                  {progress < 30 && "Analyzing social network data..."}
                  {progress >= 30 && progress < 60 && "Optimizing academic balance..."}
                  {progress >= 60 && progress < 90 && "Maximizing wellbeing metrics..."}
                  {progress >= 90 && "Finalizing classroom allocations..."}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="network" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Social Network Visualization</CardTitle>
              <CardDescription>Interactive visualization of student social networks and relationships.</CardDescription>
            </CardHeader>
            <CardContent className="h-[500px] pt-6">
              <NetworkVisualization />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Allocation Results</CardTitle>
              <CardDescription>Review and fine-tune the generated classroom allocations.</CardDescription>
            </CardHeader>
            <CardContent>
              <ClassroomAllocation />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </DashboardShell>
  )
}
