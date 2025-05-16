document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");

  form.onsubmit = async function (e) {
    e.preventDefault();

    const formData = new FormData(form);
    const academicWeight = document.getElementById("academic-weight").value;
    const wellbeingWeight = document.getElementById("wellbeing-weight").value;
    const numClasses = document.querySelector(
      'input[name="num_classes"]'
    ).value;

    formData.set("academic_weight", academicWeight);
    formData.set("wellbeing_weight", wellbeingWeight);
    formData.set("num_classes", numClasses);

    // 🚀 Show progress bar before sending request
    document.getElementById("progressBar").style.display = "block";

    try {
      const response = await fetch("/allocate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        alert("Error occurred while allocating students.");
        document.getElementById("progressBar").style.display = "none";
        return;
      }

      const data = await response.json();
      const graph1URL = data.graph_image_url;
      const graph2URL = data.graph_image2_url;

      document.getElementById("resultDiv").style.display = "block";
      document.getElementById("progressBar").style.display = "none";

      const selector = document.getElementById("class-selector");
      selector.innerHTML = "";
      data.unique_classes_random_allocated.forEach((classId) => {
        const option = document.createElement("option");
        option.value = classId;
        option.textContent = `Class ${classId}`;
        selector.appendChild(option);
      });

      document.getElementById(
        "graph-image"
      ).src = `${graph1URL}?t=${Date.now()}`;
      document.getElementById(
        "graph-image2"
      ).src = `${graph2URL}?t=${Date.now()}`;

      document
        .getElementById("class-selector")
        .addEventListener("change", async function () {
          const classId = this.value;
          const response = await fetch(`/class_graph/${classId}`);
          const data = await response.json();

          document.getElementById("class-graph-image").src = `${
            data.allocated_graph_url
          }?t=${Date.now()}`;
          document.getElementById("class-graph-image-random").src = `${
            data.random_graph_url
          }?t=${Date.now()}`;
        });
    } catch (error) {
      console.error("Allocation failed:", error);
      alert("Unexpected error occurred.");
      document.getElementById("progressBar").style.display = "none";
    }
  };

  // 🖼️ Smooth fade-in when images load
  document.querySelectorAll("img").forEach((img) => {
    if (img.complete) {
      img.classList.add("loaded");
    } else {
      img.addEventListener("load", () => {
        img.classList.add("loaded");
      });
    }
  });

  // 🧾 CSV Preview Logic
  document
    .querySelector('input[type="file"]')
    .addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function (event) {
        const text = event.target.result;
        const lines = text.split("\n").slice(0, 6); // First row + 5 rows
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
});
