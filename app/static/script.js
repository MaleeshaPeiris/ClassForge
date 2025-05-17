document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");

  form.onsubmit = async function (e) {
    e.preventDefault();

    const formData = new FormData(form);
    formData.set(
      "academic_weight",
      document.getElementById("academic-weight").value
    );
    formData.set(
      "wellbeing_weight",
      document.getElementById("wellbeing-weight").value
    );
    formData.set(
      "num_classes",
      document.querySelector('input[name="num_classes"]').value
    );

    document.getElementById("progressBar").style.display = "block";

    try {
      const response = await fetch("/allocate", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Allocation failed");

      const data = await response.json();
      const students = data.students;

      document.getElementById("resultDiv").style.display = "block";

      const counts = await (await fetch("/class_counts")).json();

      if (window.optimalChart) window.optimalChart.destroy();
      if (window.randomChart) window.randomChart.destroy();

      window.optimalChart = new Chart(
        document.getElementById("chart-optimal").getContext("2d"),
        {
          type: "bar",
          data: {
            labels: Object.keys(counts.optimal),
            datasets: [
              {
                label: "Optimal Allocation",
                data: Object.values(counts.optimal),
                backgroundColor: "#007bff",
              },
            ],
          },
          options: {
            responsive: true,
            plugins: {
              title: { display: true, text: "Optimal Class Allocation" },
              tooltip: { mode: "index", intersect: false },
            },
            scales: { y: { beginAtZero: true } },
          },
        }
      );

      window.randomChart = new Chart(
        document.getElementById("chart-random").getContext("2d"),
        {
          type: "bar",
          data: {
            labels: Object.keys(counts.random),
            datasets: [
              {
                label: "Random Allocation",
                data: Object.values(counts.random),
                backgroundColor: "#28a745",
              },
            ],
          },
          options: {
            responsive: true,
            plugins: {
              title: { display: true, text: "Random Class Allocation" },
              tooltip: { mode: "index", intersect: false },
            },
            scales: { y: { beginAtZero: true } },
          },
        }
      );

      const selector = document.getElementById("class-selector");
      selector.innerHTML = "<option disabled selected>Select Class</option>";
      data.unique_classes_random_allocated.forEach((classId) => {
        const option = document.createElement("option");
        option.value = classId;
        option.textContent = `Class ${classId}`;
        selector.appendChild(option);
      });

      selector.addEventListener("change", async function () {
        const classId = this.value;

        const graphData = await (await fetch(`/class_graph/${classId}`)).json();
        document.getElementById("class-graph-image").src = `${
          graphData.allocated_graph_url
        }?t=${Date.now()}`;
        document.getElementById("class-graph-image-random").src = `${
          graphData.random_graph_url
        }?t=${Date.now()}`;
        document.getElementById("class-id-label").textContent = classId;

        const tableData = await (
          await fetch(`/class_students/${classId}`)
        ).json();
        const tbody = document.getElementById("class-details-body");
        tbody.innerHTML = "";

        tableData.students.forEach((s) => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td>${s.student_id}</td>
            <td>${s.optimal_class}</td>
            <td>${s.random_class}</td>
            <td>${s.bully}</td>
            <td>${s.gender}</td>
          `;
          tbody.appendChild(row);
        });

        document.getElementById("class-details-table").style.display = "block";
      });

      render3DGraph(students);
      document.getElementById("progressBar").style.display = "none";
    } catch (error) {
      console.error(error);
      alert("Unexpected error occurred.");
      document.getElementById("progressBar").style.display = "none";
    }
  };

  document
    .querySelector('input[type="file"]')
    .addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function (event) {
        const text = event.target.result;
        const lines = text.split("\n").slice(0, 6);
        const table = document.getElementById("csv-table");
        table.innerHTML = "";

        lines.forEach((line, i) => {
          const row = document.createElement("tr");
          line.split(",").forEach((cell) => {
            const cellElem = document.createElement(i === 0 ? "th" : "td");
            cellElem.textContent = cell;
            row.appendChild(cellElem);
          });
          table.appendChild(row);
        });

        document.getElementById("csv-preview").style.display = "block";
      };
      reader.readAsText(file);
    });

  document.querySelectorAll("img").forEach((img) => {
    if (img.complete) {
      img.classList.add("loaded");
    } else {
      img.addEventListener("load", () => img.classList.add("loaded"));
    }
  });
});

// 🌐 3D Graph with Class Filter Buttons and Smart Layout
function render3DGraph(students) {
  const container = document.getElementById("graph-3d");
  const buttonContainer = document.getElementById("button-container");
  const classButtonsDiv = document.getElementById("class-buttons");

  if (!container) return;
  container.innerHTML = "";
  buttonContainer.innerHTML = "";
  classButtonsDiv.style.display = "block";

  const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

  const nodes = students.map((s, i) => ({
    id: i,
    label: s.student_id,
    class: s.optimal_class,
    color: colorScale(s.optimal_class),
  }));

  const links = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    links.push({ source: i, target: i + 1 });
  }

  const fullData = { nodes, links };

  function filterGraph(classId) {
    const visibleNodes =
      classId === "all" ? nodes : nodes.filter((n) => n.class == classId);

    const nodeMap = new Map(visibleNodes.map((n) => [n.id, n]));
    let visibleLinks = [];

    if (classId === "all") {
      for (let i = 0; i < visibleNodes.length - 1; i++) {
        visibleLinks.push({
          source: visibleNodes[i],
          target: visibleNodes[i + 1],
        });
      }
    } else {
      for (let i = 0; i < visibleNodes.length; i++) {
        for (let j = i + 1; j < visibleNodes.length; j++) {
          visibleLinks.push({
            source: visibleNodes[i],
            target: visibleNodes[j],
          });
        }
      }
    }

    window.FORCE_GRAPH_INSTANCE.graphData({
      nodes: visibleNodes,
      links: visibleLinks,
    });

    visibleNodes.forEach((n) => {
      n.z = (Math.random() - 0.5) * 300;
    });

    setTimeout(() => {
      window.FORCE_GRAPH_INSTANCE.zoomToFit(800, 100);
    }, 400);
  }

  const uniqueClasses = [...new Set(nodes.map((n) => n.class))].sort();

  const btnAll = document.createElement("button");
  btnAll.className = "btn btn-outline-primary m-1";
  btnAll.textContent = "All";
  btnAll.onclick = () => filterGraph("all");
  buttonContainer.appendChild(btnAll);

  uniqueClasses.forEach((cls) => {
    const btn = document.createElement("button");
    btn.className = "btn btn-outline-secondary m-1";
    btn.textContent = `Class ${cls}`;
    btn.onclick = () => filterGraph(cls);
    buttonContainer.appendChild(btn);
  });

  const Graph = ForceGraph3D()(container)
    .graphData(fullData)
    .nodeLabel((d) => `🎓 ${d.label}<br>🏫 Class: ${d.class}`)
    .nodeAutoColorBy("class")
    .linkColor(() => "#404040")
    .linkOpacity(0.3)
    .linkWidth(7)
    .backgroundColor("#FFF")
    .cooldownTicks(300)
    .d3Force("link", d3.forceLink().distance(300))
    .d3Force("center", d3.forceCenter(0, 0, 0))
    .onEngineStop(() => Graph.zoomToFit(800, 100));

  window.FORCE_GRAPH_INSTANCE = Graph;

  Graph.cameraPosition({ x: 0, y: 0, z: 320 }, { x: 0, y: 0, z: 0 }, 2000);

  Graph.nodeThreeObject((node) => {
    const geometry = new THREE.SphereGeometry(20, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: node.color });
    return new THREE.Mesh(geometry, material);
  });

  const resize = () => {
    Graph.width(container.clientWidth);
    Graph.height(container.clientHeight);
  };

  window.addEventListener("resize", resize);
  resize();
}
