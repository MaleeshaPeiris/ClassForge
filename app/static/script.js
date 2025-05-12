document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");

    form.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(form);
        const response = await fetch("/allocate", {
            method: "POST",
            body: formData
        });

        const students = await response.json();
        const output = document.getElementById("output");
        const ctx = document.getElementById("chart").getContext("2d");

        // Count students per class
        const classCounts = {};
        students.forEach(s => {
            const cls = s.allocated_class;
            classCounts[cls] = (classCounts[cls] || 0) + 1;
        });

        const classLabels = Object.keys(classCounts);
        const classValues = Object.values(classCounts);

        // Clear any existing chart
        if (window.barChart) window.barChart.destroy();

        // Create new bar chart
        window.barChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: classLabels,
                datasets: [{
                    label: "Students per Class",
                    data: classValues,
                    backgroundColor: "rgba(54, 162, 235, 0.5)",
                    borderColor: "rgba(54, 162, 235, 1)",
                    borderWidth: 1
                }]
            },
            options: {
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
            }
        });
    };
});
