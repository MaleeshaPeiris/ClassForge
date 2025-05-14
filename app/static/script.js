document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");

    form.onsubmit = async function (e) {
        e.preventDefault();

        const formData = new FormData(form);

        // Append slider values explicitly (in case JavaScript handles them outside form scope)
        const academicWeight = document.getElementById("academic-weight").value;
        const wellbeingWeight = document.getElementById("wellbeing-weight").value;
        const numClasses = document.querySelector('input[name="num_classes"]').value;

        formData.set("academic_weight", academicWeight);
        formData.set("wellbeing_weight", wellbeingWeight);
        formData.set("num_classes", numClasses); // ensure it's properly updated

        const response = await fetch("/allocate", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            alert("Error occurred while allocating students.");
            return;
        }

        const data = await response.json();
        const students = data.students;
        const graph1URL = data.graph_image_url;
        const graph2URL = data.graph_image2_url; 
        const output = document.getElementById("output");


        // Count students per class
        const classCounts = {};
        students.forEach(s => {
            const cls = s.random_label;
            classCounts[cls] = (classCounts[cls] || 0) + 1;
        });

        const classLabels = Object.keys(classCounts);
        const classValues = Object.values(classCounts);

        // Clear previous charts
        if (window.barChart1) window.barChart1.destroy();
        if (window.barChart2) window.barChart2.destroy();

        // Chart data setup (same for both)
        const barData = {
            labels: classLabels,
            datasets: [{
                label: "Students per Class",
                data: classValues,
                backgroundColor: "rgba(54, 162, 235, 0.5)",
                borderColor: "rgba(54, 162, 235, 1)",
                borderWidth: 1
            }]
        };

        // Chart options (same unless you want to customize)
        const barOptions = {
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const classClicked = classLabels[index];
                    const filtered = students.filter(s => s.allocated_class == classClicked);

                    const detailDiv = document.getElementById("student-details");
                    detailDiv.innerHTML = `<h3>Class ${classClicked} - Students</h3><pre>${JSON.stringify(filtered, null, 2)}</pre>`;
                }
            },
            scales: {
                y: { beginAtZero: true }
            }
        };

        // Chart 1
        const ctx1 = document.getElementById("chart1").getContext("2d");
        window.barChart1 = new Chart(ctx1, {
            type: "bar",
            data: barData,
            options: barOptions
        });

        // Chart 2 (duplicate)
        const ctx2 = document.getElementById("chart2").getContext("2d");
        window.barChart2 = new Chart(ctx2, {
            type: "bar",
            data: barData,
            options: barOptions
        });
        
        document.getElementById("graph-image").src = `${graph1URL}?t=${new Date().getTime()}`;
        document.getElementById("graph-image2").src = `${graph2URL}?t=${new Date().getTime()}`;
        
    };
});
