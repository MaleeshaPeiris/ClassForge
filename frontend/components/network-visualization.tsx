// "use client"

// import { useEffect, useRef, useState } from "react"
// import * as d3 from "d3"
// import { Button } from "@/components/ui/button"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// // Sample data - in a real app, this would come from an API
// const generateNetworkData = () => {
//   const nodes = []
//   const links = []

//   // Generate 100 student nodes
//   for (let i = 0; i < 100; i++) {
//     nodes.push({
//       id: `student-${i}`,
//       name: `Student ${i}`,
//       group: Math.floor(i / 25), // Assign to one of 4 groups
//       academicScore: 50 + Math.random() * 50,
//       wellbeingScore: 50 + Math.random() * 50,
//     })
//   }

//   // Generate friendship links (more dense within groups)
//   for (let i = 0; i < nodes.length; i++) {
//     const groupId = nodes[i].group

//     // Create in-group connections (higher probability)
//     for (let j = 0; j < nodes.length; j++) {
//       if (i !== j && nodes[j].group === groupId && Math.random() > 0.7) {
//         links.push({
//           source: nodes[i].id,
//           target: nodes[j].id,
//           type: "friendship",
//           strength: 0.5 + Math.random() * 0.5,
//         })
//       }
//     }

//     // Create some out-group connections (lower probability)
//     for (let j = 0; j < nodes.length; j++) {
//       if (i !== j && nodes[j].group !== groupId && Math.random() > 0.95) {
//         links.push({
//           source: nodes[i].id,
//           target: nodes[j].id,
//           type: "friendship",
//           strength: 0.1 + Math.random() * 0.4,
//         })
//       }
//     }

//     // Create some negative interactions
//     if (Math.random() > 0.9) {
//       const targetIndex = Math.floor(Math.random() * nodes.length)
//       if (i !== targetIndex) {
//         links.push({
//           source: nodes[i].id,
//           target: nodes[targetIndex].id,
//           type: "negative",
//           strength: 0.1 + Math.random() * 0.3,
//         })
//       }
//     }
//   }

//   return { nodes, links }
// }

// const networkData = generateNetworkData()

// interface D3NetworkProps {
//   data: {
//     nodes: any[]
//     links: any[]
//   }
//   width: number
//   height: number
//   networkType: string
// }

// function D3Network({ data, width, height, networkType }: D3NetworkProps) {
//   const svgRef = useRef<SVGSVGElement>(null)
//   const tooltipRef = useRef<HTMLDivElement>(null)

//   useEffect(() => {
//     if (!svgRef.current) return

//     // Clear previous visualization
//     d3.select(svgRef.current).selectAll("*").remove()

//     // Filter links based on network type
//     const filteredLinks = networkType === "all" ? data.links : data.links.filter((link) => link.type === networkType)

//     // Create a filtered dataset
//     const filteredData = {
//       nodes: data.nodes,
//       links: filteredLinks,
//     }

//     // Set up the simulation
//     const simulation = d3
//       .forceSimulation(filteredData.nodes)
//       .force(
//         "link",
//         d3
//           .forceLink(filteredData.links)
//           .id((d: any) => d.id)
//           .distance((link) => (networkType === "negative" ? 100 : 50)),
//       )
//       .force("charge", d3.forceManyBody().strength(-100))
//       .force("center", d3.forceCenter(width / 2, height / 2))
//       .force("collide", d3.forceCollide().radius(20))

//     // Create the SVG container
//     const svg = d3
//       .select(svgRef.current)
//       .attr("width", width)
//       .attr("height", height)
//       .attr("viewBox", [0, 0, width, height])
//       .attr("style", "max-width: 100%; height: auto;")

//     // Create a group for the visualization
//     const g = svg.append("g")

//     // Add zoom functionality
//     const zoom = d3
//       .zoom()
//       .scaleExtent([0.1, 4])
//       .on("zoom", (event) => {
//         g.attr("transform", event.transform)
//       })

//     svg.call(zoom as any)

//     // Create tooltip
//     const tooltip = d3
//       .select(tooltipRef.current)
//       .style("position", "absolute")
//       .style("visibility", "hidden")
//       .style("background-color", "white")
//       .style("border", "1px solid #ddd")
//       .style("border-radius", "4px")
//       .style("padding", "8px")
//       .style("font-size", "12px")
//       .style("pointer-events", "none")
//       .style("z-index", "10")

//     // Create the links
//     const link = g
//       .append("g")
//       .attr("stroke-opacity", 0.6)
//       .selectAll("line")
//       .data(filteredData.links)
//       .join("line")
//       .attr("stroke", (d: any) => (d.type === "friendship" ? "#52c41a" : "#ff4d4f"))
//       .attr("stroke-width", (d: any) => d.strength * 3)

//     // Create the nodes
//     const node = g
//       .append("g")
//       .selectAll("circle")
//       .data(filteredData.nodes)
//       .join("circle")
//       .attr("r", 8)
//       .attr("fill", (d: any) => ["#ff4d4f", "#52c41a", "#1890ff", "#722ed1"][d.group])
//       .call(drag(simulation) as any)
//       .on("mouseover", (event, d: any) => {
//         tooltip
//           .style("visibility", "visible")
//           .html(`
//             <strong>${d.name}</strong><br/>
//             Academic: ${Math.round(d.academicScore)}<br/>
//             Wellbeing: ${Math.round(d.wellbeingScore)}
//           `)
//           .style("left", event.pageX + 10 + "px")
//           .style("top", event.pageY - 28 + "px")
//       })
//       .on("mouseout", () => {
//         tooltip.style("visibility", "hidden")
//       })

//     // Add labels to nodes
//     const labels = g
//       .append("g")
//       .selectAll("text")
//       .data(filteredData.nodes)
//       .join("text")
//       .text((d: any) => d.name)
//       .attr("font-size", "8px")
//       .attr("dx", 10)
//       .attr("dy", 4)
//       .style("pointer-events", "none")

//     // Update positions on simulation tick
//     simulation.on("tick", () => {
//       link
//         .attr("x1", (d: any) => d.source.x)
//         .attr("y1", (d: any) => d.source.y)
//         .attr("x2", (d: any) => d.target.x)
//         .attr("y2", (d: any) => d.target.y)

//       node.attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y)

//       labels.attr("x", (d: any) => d.x).attr("y", (d: any) => d.y)
//     })

//     // Drag functionality
//     function drag(simulation: any) {
//       function dragstarted(event: any) {
//         if (!event.active) simulation.alphaTarget(0.3).restart()
//         event.subject.fx = event.subject.x
//         event.subject.fy = event.subject.y
//       }

//       function dragged(event: any) {
//         event.subject.fx = event.x
//         event.subject.fy = event.y
//       }

//       function dragended(event: any) {
//         if (!event.active) simulation.alphaTarget(0)
//         event.subject.fx = null
//         event.subject.fy = null
//       }

//       return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended)
//     }

//     // Cleanup
//     return () => {
//       simulation.stop()
//     }
//   }, [data, width, height, networkType])

//   return (
//     <div className="relative">
//       <svg ref={svgRef} />
//       <div ref={tooltipRef} />
//     </div>
//   )
// }

// export function NetworkVisualization() {
//   const [networkType, setNetworkType] = useState("all")
//   const containerRef = useRef<HTMLDivElement>(null)
//   const [dimensions, setDimensions] = useState({ width: 800, height: 450 })

//   useEffect(() => {
//     if (containerRef.current) {
//       const { width, height } = containerRef.current.getBoundingClientRect()
//       setDimensions({ width, height })
//     }

//     const handleResize = () => {
//       if (containerRef.current) {
//         const { width, height } = containerRef.current.getBoundingClientRect()
//         setDimensions({ width, height })
//       }
//     }

//     window.addEventListener("resize", handleResize)
//     return () => window.removeEventListener("resize", handleResize)
//   }, [])

//   return (
//     <div className="h-full">
//       <Tabs defaultValue="all" className="mb-4" onValueChange={setNetworkType}>
//         <TabsList>
//           <TabsTrigger value="all">All Connections</TabsTrigger>
//           <TabsTrigger value="friendship">Friendship Network</TabsTrigger>
//           <TabsTrigger value="negative">Negative Interactions</TabsTrigger>
//         </TabsList>

//         <TabsContent value="all" className="h-[450px]">
//           <div ref={containerRef} className="h-full">
//             <D3Network data={networkData} width={dimensions.width} height={dimensions.height} networkType="all" />
//           </div>
//         </TabsContent>

//         <TabsContent value="friendship" className="h-[450px]">
//           <div ref={containerRef} className="h-full">
//             <D3Network
//               data={networkData}
//               width={dimensions.width}
//               height={dimensions.height}
//               networkType="friendship"
//             />
//           </div>
//         </TabsContent>

//         <TabsContent value="negative" className="h-[450px]">
//           <div ref={containerRef} className="h-full">
//             <D3Network data={networkData} width={dimensions.width} height={dimensions.height} networkType="negative" />
//           </div>
//         </TabsContent>
//       </Tabs>

//       <div className="flex space-x-2 mt-4">
//         <Button variant="outline" size="sm">
//           Export Network Data
//         </Button>
//         <Button variant="outline" size="sm">
//           Apply Network Insights
//         </Button>
//       </div>
//     </div>
//   )
// }

// // version 2

// "use client"

// import { useEffect, useRef, useState } from "react"
// import * as d3 from "d3"
// import { Button } from "@/components/ui/button"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// // =====================
// // Define data interfaces
// // =====================
// interface Node extends d3.SimulationNodeDatum {
//   id: number
//   group: number
//   gpa: number
//   score: number
//   motivation: number
//   education_level: number
//   fx?: number | null
//   fy?: number | null
// }

// interface Link extends d3.SimulationLinkDatum<Node> {
//   source: number | Node
//   target: number | Node
//   strength: number
//   type: string // e.g., "friendship" or "negative"
// }

// interface GraphData {
//   nodes: Node[]
//   links: Link[]
// }

// // =====================
// // D3 Network Graph Component
// // =====================
// function D3Network({ data, width, height, networkType }: { data: GraphData; width: number; height: number; networkType: string }) {
//   const svgRef = useRef<SVGSVGElement>(null)
//   const tooltipRef = useRef<HTMLDivElement>(null)

//   useEffect(() => {
//     if (!svgRef.current) return

//     const svg = d3.select(svgRef.current)
//     svg.selectAll("*").remove() // Clear previous render

//     // Fix: Normalize link source/target to node IDs
//     const fixedLinks = data.links.map((link) => ({
//       ...link,
//       source: typeof link.source === 'object' ? (link.source as Node).id : link.source,
//       target: typeof link.target === 'object' ? (link.target as Node).id : link.target
//     }))

//     // Filter links if needed
//     const filteredLinks = networkType === "all" ? fixedLinks : fixedLinks.filter((link) => link.type === networkType)

//     // Set up D3 force simulation
//     const simulation = d3.forceSimulation<Node>(data.nodes)
//       .force("link", d3.forceLink<Node, Link>(filteredLinks)
//         .id((d) => d.id.toString()) // match by node ID
//         .distance(100)
//       )
//       .force("charge", d3.forceManyBody<Node>().strength(-50))
//       .force("center", d3.forceCenter(width / 2, height / 2))
//       .force("collide", d3.forceCollide<Node>().radius(20))

//     const g = svg.append("g") // Group for zooming

//     // Tooltip setup
//     const tooltip = tooltipRef.current
//       ? d3.select<HTMLDivElement, unknown>(tooltipRef.current)
//           .style("position", "absolute")
//           .style("visibility", "hidden")
//           .style("background-color", "white")
//           .style("border", "1px solid #ddd")
//           .style("border-radius", "4px")
//           .style("padding", "8px")
//           .style("font-size", "12px")
//           .style("pointer-events", "none")
//           .style("z-index", "10")
//       : null

//     // Draw links
//     g.selectAll<SVGLineElement, Link>("line")
//       .data(filteredLinks)
//       .enter()
//       .append("line")
//       .attr("stroke", (d) => (d.type === "friendship" ? "#52c41a" : "#ff4d4f"))
//       .attr("stroke-width", (d) => d.strength * 3)

//     // Draw nodes with drag
//     g.selectAll<SVGCircleElement, Node>("circle")
//       .data(data.nodes)
//       .enter()
//       .append("circle")
//       .attr("r", 8)
//       .attr("fill", (d) => ["#ff4d4f", "#52c41a", "#1890ff", "#722ed1"][d.group % 4])
//       .call(
//         d3.drag<SVGCircleElement, Node>()
//           .on("start", (event, d) => {
//             if (!event.active) simulation.alphaTarget(0.3).restart()
//             d.fx = d.x
//             d.fy = d.y
//           })
//           .on("drag", (event, d) => {
//             d.fx = event.x
//             d.fy = event.y
//           })
//           .on("end", (event, d) => {
//             if (!event.active) simulation.alphaTarget(0)
//             d.fx = null
//             d.fy = null
//           })
//       )
//       .on("mouseover", (event, d) => {
//         tooltip?.style("visibility", "visible")
//           .html(`
//             <strong>ID: ${d.id}</strong><br/>
//             Group: ${d.group}<br/>
//             GPA: ${d.gpa}<br/>
//             Score: ${d.score}<br/>
//             Motivation: ${d.motivation}<br/>
//             Education Level: ${d.education_level}
//           `)
//           .style("left", event.pageX + 10 + "px")
//           .style("top", event.pageY - 28 + "px")
//       })
//       .on("mouseout", () => {
//         tooltip?.style("visibility", "hidden")
//       })

//     // Add labels to nodes
//     g.selectAll<SVGTextElement, Node>("text")
//       .data(data.nodes)
//       .enter()
//       .append("text")
//       .text((d) => `ID:${d.id}`)
//       .attr("font-size", "8px")
//       .attr("dx", 10)
//       .attr("dy", 4)
//       .style("pointer-events", "none")

//     // Update node positions on each tick
//     simulation.on("tick", () => {
//       g.selectAll<SVGLineElement, Link>("line")
//         .attr("x1", (d) => (typeof d.source === "object" ? d.source.x! : 0))
//         .attr("y1", (d) => (typeof d.source === "object" ? d.source.y! : 0))
//         .attr("x2", (d) => (typeof d.target === "object" ? d.target.x! : 0))
//         .attr("y2", (d) => (typeof d.target === "object" ? d.target.y! : 0))

//       g.selectAll<SVGCircleElement, Node>("circle")
//         .attr("cx", (d) => d.x!)
//         .attr("cy", (d) => d.y!)

//       g.selectAll<SVGTextElement, Node>("text")
//         .attr("x", (d) => d.x!)
//         .attr("y", (d) => d.y!)
//     })

//     // Setup zoom
//     const zoom = d3.zoom<SVGSVGElement, unknown>()
//       .scaleExtent([0.1, 4])
//       .on("zoom", (event) => {
//         g.attr("transform", event.transform.toString())
//       })

//     svg.call(zoom)

//     // Cleanup on component unmount
//     return () => {
//       simulation.stop()
//     }
//   }, [data, width, height, networkType])

//   return (
//     <div className="relative">
//       <svg ref={svgRef} width={width} height={height} />
//       <div ref={tooltipRef} />
//     </div>
//   )
// }


// // =====================
// // Wrapper component for Tabs and Graph
// // =====================
// export function NetworkVisualization() {
//   const [networkType, setNetworkType] = useState("all")
//   const containerRef = useRef<HTMLDivElement>(null)
//   const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
//   const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] })
//   const [loading, setLoading] = useState(true)
//   const [error, setError] = useState<string | null>(null)

//   useEffect(() => {
//     async function fetchData() {
//       setLoading(true)
//       setError(null)
//       try {
//         const response = await fetch("http://localhost:5000/api/graph")
//         if (!response.ok) throw new Error("Failed to fetch network data")
//         const data = await response.json()
//         setGraphData(data)
//       } catch (error: any) {
//         setError(error.message)
//       } finally {
//         setLoading(false)
//       }
//     }
//     fetchData()
//   }, [])

//   useEffect(() => {
//     if (containerRef.current) {
//       const { width, height } = containerRef.current.getBoundingClientRect()
//       setDimensions({ width, height })
//     }
//   }, [])

//   // 🆕 ADD THIS FUNCTION RIGHT HERE
//   function handleExportGraph(data: GraphData) {
//     const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
//       JSON.stringify(data, null, 2)
//     )}`;
//     const link = document.createElement("a");
//     link.href = jsonString;
//     link.download = "network_graph.json";
//     link.click();
//   }

//   // ✅ Then continue to your normal loading / error checks
//   if (loading) return <div>Loading network...</div>
//   if (error) return <div>Error: {error}</div>

//   return (
//     <div className="h-full">
//       <Tabs defaultValue="all" className="mb-4" onValueChange={setNetworkType}>
//         <TabsList>
//           <TabsTrigger value="all">All Connections</TabsTrigger>
//           <TabsTrigger value="friendship">Friendship Network</TabsTrigger>
//           <TabsTrigger value="negative">Negative Interactions</TabsTrigger>
//         </TabsList>

//         {/* Your D3Network rendering tabs */}

//         <div className="flex space-x-2 mt-4">
//           <Button variant="outline" size="sm" onClick={() => handleExportGraph(graphData)}>
//             Export Network Data
//           </Button>
//           <Button variant="outline" size="sm">
//             Apply Network Insights
//           </Button>
//         </div>
//       </Tabs>
//     </div>
//   )
// }

// version 3

"use client"

import { useEffect, useRef, useState } from "react"
import * as d3 from "d3"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"


interface Node extends d3.SimulationNodeDatum {
  id: number
  group: number
  gpa: number
  score: number
  motivation: number
  education_level: number
}

interface Link {
  source: number
  target: number
  strength: number
  type: string // Added type for filtering
}

interface GraphData {
  nodes: Node[]
  links: Link[]
}

function D3Network({ data, width, height, networkType }: { data: GraphData; width: number; height: number; networkType: string }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    // Filter links by type if not "all"
    const filteredLinks = networkType === "all" ? data.links : data.links.filter((link) => link.type === networkType)

    // Use D3 forceLink with .id() so source/target become node objects
    const simulation = d3.forceSimulation(data.nodes)
      .force("link", d3.forceLink(filteredLinks).id((d: any) => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-50))
      .force("center", d3.forceCenter(width / 2, height / 2))

    const g = svg.append("g")

    // Tooltip
    let tooltip: d3.Selection<HTMLDivElement, unknown, null, undefined> | null = null
    if (tooltipRef.current) {
      tooltip = d3.select(tooltipRef.current)
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background-color", "white")
        .style("border", "1px solid #ddd")
        .style("border-radius", "4px")
        .style("padding", "8px")
        .style("font-size", "12px")
        .style("pointer-events", "none")
        .style("z-index", "10")
    }

    g.selectAll("line")
      .data(filteredLinks)
      .enter()
      .append("line")
      .attr("stroke", (d: any) => d.type === "friendship" ? "#52c41a" : "#ff4d4f")
      .attr("stroke-width", (d: any) => d.strength * 3)

    const node = g.selectAll("circle")
      .data(data.nodes)
      .enter()
      .append("circle")
      .attr("r", 8)
      .attr("fill", (d) => ["#ff4d4f", "#52c41a", "#1890ff", "#722ed1"][d.group % 4])
      .call(d3.drag()
        .on("start", (event, d: any) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on("drag", (event, d: any) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on("end", (event, d: any) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      )
      .on("mouseover", (event, d: any) => {
        if (tooltip) {
          tooltip
            .style("visibility", "visible")
            .html(`
              <strong>ID: ${d.id}</strong><br/>
              Group: ${d.group}<br/>
              GPA: ${d.gpa}<br/>
              Score: ${d.score}<br/>
              Motivation: ${d.motivation}<br/>
              Education Level: ${d.education_level}
            `)
            .style("left", event.pageX + 10 + "px")
            .style("top", event.pageY - 28 + "px")
        }
      })
      .on("mouseout", () => {
        if (tooltip) tooltip.style("visibility", "hidden")
      })

    // Add labels
    g.selectAll("text")
      .data(data.nodes)
      .enter()
      .append("text")
      .text((d: any) => `ID:${d.id}`)
      .attr("font-size", "8px")
      .attr("dx", 10)
      .attr("dy", 4)
      .style("pointer-events", "none")

    simulation.on("tick", () => {
      g.selectAll("line")
        .attr("x1", (d: any) => typeof d.source === "object" ? d.source.x : 0)
        .attr("y1", (d: any) => typeof d.source === "object" ? d.source.y : 0)
        .attr("x2", (d: any) => typeof d.target === "object" ? d.target.x : 0)
        .attr("y2", (d: any) => typeof d.target === "object" ? d.target.y : 0)

      g.selectAll("circle")
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y)

      g.selectAll("text")
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y)
    })
  }, [data, width, height, networkType])

  // Show message if no nodes or links
  if (!data.nodes.length || !data.links.length) {
    return <div>No network data to display.</div>
  }

  return (
    <div className="relative">
      <svg ref={svgRef} width={width} height={height} />
      <div ref={tooltipRef} />
    </div>
  )
}


export function NetworkVisualization() {
  const [networkType, setNetworkType] = useState("all")
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      setLoading(true)
      setError(null)
      try {
        const response = await fetch("http://localhost:5000/api/graph")
        if (!response.ok) throw new Error("Failed to fetch network data")
        const data = await response.json()
        setGraphData(data)
      } catch (error: any) {
        setError(error.message)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) return <div>Loading network...</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div className="h-full">
      <Tabs defaultValue="all" className="mb-4" onValueChange={setNetworkType}>
        <TabsList>
          <TabsTrigger value="all">All Connections</TabsTrigger>
          <TabsTrigger value="friendship">Friendship Network</TabsTrigger>
          <TabsTrigger value="negative">Negative Interactions</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="h-[600px]">
          <div ref={containerRef} className="h-full">
            <D3Network data={graphData} width={dimensions.width} height={dimensions.height} networkType="all" />
          </div>
        </TabsContent>
        <TabsContent value="friendship" className="h-[600px]">
          <div ref={containerRef} className="h-full">
            <D3Network data={graphData} width={dimensions.width} height={dimensions.height} networkType="friendship" />
          </div>
        </TabsContent>
        <TabsContent value="negative" className="h-[600px]">
          <div ref={containerRef} className="h-full">
            <D3Network data={graphData} width={dimensions.width} height={dimensions.height} networkType="negative" />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}