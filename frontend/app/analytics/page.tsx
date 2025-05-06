import { DashboardHeader } from "@/components/dashboard-header"
import { DashboardShell } from "@/components/dashboard-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { NetworkVisualization } from "@/components/network-visualization"

export default function AnalyticsPage() {
  return (
    <DashboardShell>
      <DashboardHeader heading="Network Analytics" text="Advanced social network analysis and visualization tools." />

      <Tabs defaultValue="network" className="space-y-4">
        <TabsList>
          <TabsTrigger value="network">Network Visualization</TabsTrigger>
          <TabsTrigger value="metrics">Network Metrics</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
        </TabsList>

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

        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Network Metrics</CardTitle>
              <CardDescription>Key metrics about the social network structure.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center bg-muted rounded-md">
                <p className="text-muted-foreground">Network metrics visualization will appear here.</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI-Generated Network Insights</CardTitle>
              <CardDescription>Automatically generated insights from social network analysis.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-md">
                  <h3 className="font-medium">Strong Friendship Clusters</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Analysis has identified 5 strong friendship clusters that should be preserved in classroom
                    allocations. These clusters show high academic collaboration and positive social support.
                  </p>
                </div>

                <div className="p-4 border rounded-md">
                  <h3 className="font-medium">Potential Conflict Areas</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    3 areas of potential conflict have been detected, involving 12 students across multiple groups.
                    Recommended to separate these students in classroom allocations.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </DashboardShell>
  )
}