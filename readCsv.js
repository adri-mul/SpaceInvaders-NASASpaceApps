document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('csvFileInput');
    const output = document.getElementById('output');

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();

        reader.onload = function(e) {
            const text = e.target.result;
            let csvArray = csvToArray(text);

            // Ignore lines starting with #
            csvArray = csvArray.filter(row => {
                const firstCell = row[0]?.trim();
                return !firstCell || firstCell[0] !== '#';
            });

            // Convert each row to a JSON object
            const jsonObjects = csvArray.map(row => ({
                source: "User",
                id: row[1] || "",
                display: row[1] || "",
                search_query: ""
            }));

            // Display JSON on page
            output.textContent = JSON.stringify(jsonObjects, null, 2);

            // Create a downloadable JSON file
            saveJSONFile(jsonObjects, 'data.json');
        };

        reader.readAsText(file);
    });
});

// Split CSV text into array of arrays
function csvToArray(csvText) {
    const rows = csvText.trim().split('\n');
    return rows.map(row => row.split(','));
}

// Save JSON objects as a downloadable file
function saveJSONFile(jsonObjects, filename) {
    const blob = new Blob([JSON.stringify(jsonObjects, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
