"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function AllocationParameters() {
  const [academicBalance, setAcademicBalance] = useState(70)
  const [wellbeingDistribution, setWellbeingDistribution] = useState(80)
  const [friendshipRetention, setFriendshipRetention] = useState(60)
  const [behavioralConsiderations, setBehavioralConsiderations] = useState(50)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Allocation Parameters</CardTitle>
        <CardDescription>Configure the weighting of different factors for classroom allocation.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <Tabs defaultValue="weights" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="weights">Factor Weights</TabsTrigger>
            <TabsTrigger value="constraints">Constraints</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="weights" className="space-y-6 pt-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="academic-balance">Academic Balance</Label>
                <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground">
                  {academicBalance}%
                </span>
              </div>
              <Slider
                id="academic-balance"
                min={0}
                max={100}
                step={1}
                value={[academicBalance]}
                onValueChange={(value) => setAcademicBalance(value[0])}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <p className="text-xs text-muted-foreground">
                Prioritize even distribution of academic performance across classrooms.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="wellbeing-distribution">Wellbeing Distribution</Label>
                <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground">
                  {wellbeingDistribution}%
                </span>
              </div>
              <Slider
                id="wellbeing-distribution"
                min={0}
                max={100}
                step={1}
                value={[wellbeingDistribution]}
                onValueChange={(value) => setWellbeingDistribution(value[0])}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <p className="text-xs text-muted-foreground">Balance student wellbeing indicators across classrooms.</p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="friendship-retention">Friendship Retention</Label>
                <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground">
                  {friendshipRetention}%
                </span>
              </div>
              <Slider
                id="friendship-retention"
                min={0}
                max={100}
                step={1}
                value={[friendshipRetention]}
                onValueChange={(value) => setFriendshipRetention(value[0])}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <p className="text-xs text-muted-foreground">
                Maintain positive friendship connections within classrooms.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="behavioral-considerations">Behavioral Considerations</Label>
                <span className="w-12 rounded-md border border-transparent px-2 py-0.5 text-right text-sm text-muted-foreground">
                  {behavioralConsiderations}%
                </span>
              </div>
              <Slider
                id="behavioral-considerations"
                min={0}
                max={100}
                step={1}
                value={[behavioralConsiderations]}
                onValueChange={(value) => setBehavioralConsiderations(value[0])}
                className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4"
              />
              <p className="text-xs text-muted-foreground">
                Minimize disruptive behavior by separating negative interactions.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="constraints" className="space-y-6 pt-4">
            <div className="flex items-center space-x-2">
              <Switch id="max-class-size" />
              <Label htmlFor="max-class-size">Maximum class size: 30 students</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch id="gender-balance" defaultChecked />
              <Label htmlFor="gender-balance">Maintain gender balance</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch id="special-needs" defaultChecked />
              <Label htmlFor="special-needs">Distribute students with special needs</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch id="separate-conflicts" defaultChecked />
              <Label htmlFor="separate-conflicts">Separate students with conflict history</Label>
            </div>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6 pt-4">
            <div className="flex items-center space-x-2">
              <Switch id="use-reinforcement-learning" />
              <Label htmlFor="use-reinforcement-learning">Use reinforcement learning</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch id="use-gnn" defaultChecked />
              <Label htmlFor="use-gnn">Apply Graph Neural Networks</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch id="use-genetic-algorithm" defaultChecked />
              <Label htmlFor="use-genetic-algorithm">Use genetic algorithms</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch id="enable-nlp" />
              <Label htmlFor="enable-nlp">Enable NLP for teacher comments</Label>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
