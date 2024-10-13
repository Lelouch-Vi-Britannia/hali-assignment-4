document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResults(data);
        displayChart(data);
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        docDiv.innerHTML = `<strong>Document ${data.indices[i]}</strong><p>${data.documents[i]}</p><br><strong>Similarity: ${data.similarities[i]}</strong>`;
        resultsDiv.appendChild(docDiv);
    }
}

function displayChart(data) {
    // Input: data (object) - contains the following keys:
    //        - documents (list) - list of documents
    //        - indices (list) - list of indices   
    //        - similarities (list) - list of similarities
    // TODO: Implement function to display chart here
    //       There is a canvas element in the HTML file with the id 'similarity-chart'
    let ctx = document.getElementById('similarity-chart').getContext('2d');

    // Destroy the previous chart instance if it exists (to avoid overlap)
    if (window.similarityChart) {
        window.similarityChart.destroy();
    }

    // Create a new chart instance
    window.similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.indices.map(index => `Doc ${index}`),  // Labels will be the document indices
            datasets: [{
                label: 'Cosine Similarity',
                data: data.similarities,  // Similarity values for the top 5 documents
                backgroundColor: 'rgba(54, 162, 235, 0.6)',  // Light blue color for the bars
                borderColor: 'rgba(54, 162, 235, 1)',  // Darker blue for the borders
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,  // Ensure the y-axis starts at 0
                    max: 1  // Cosine similarity ranges from 0 to 1
                }
            },
            responsive: true,
            plugins: {
                legend: {
                    display: false  // Hide the legend since we only have one dataset
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Similarity: ${context.raw.toFixed(4)}`;  // Show similarity with 4 decimal points
                        }
                    }
                }
            }
        }
    });
}