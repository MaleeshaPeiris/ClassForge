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

        // Dynamically populate class dropdown
        const selector = document.getElementById("class-selector");
        selector.innerHTML = ""; // Clear old options
        data.unique_classes_random_allocated.forEach(classId => {
            const option = document.createElement("option");
            option.value = classId;
            option.textContent = `Class ${classId}`;
            selector.appendChild(option);
        });


        document.getElementById("class-selector").addEventListener("change", async function () {
            const classId = this.value;
            const response = await fetch(`/class_graph/${classId}`);
            const data = await response.json();
            document.getElementById("class-graph-image").src = data.allocated_graph_url + `?t=${new Date().getTime()}`;
            document.getElementById("class-graph-image-random").src = data.random_graph_url + `?t=${new Date().getTime()}`;
        });

        
        document.getElementById("graph-image").src = `${graph1URL}?t=${new Date().getTime()}`;
        document.getElementById("graph-image2").src = `${graph2URL}?t=${new Date().getTime()}`;
        
    };
});
