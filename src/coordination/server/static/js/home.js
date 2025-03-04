document.addEventListener('DOMContentLoaded', function() {
    // Elements for form control
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    const stopTrainingBtn = document.getElementById('stopTrainingBtn');
    const trainingForm = document.getElementById('trainingForm');
    const cancelTrainingBtn = document.getElementById('cancelTrainingBtn');
    const newTrainingForm = document.getElementById('newTrainingForm');
    let accuracyChart = null;
    

    // Show training form
    startTrainingBtn.addEventListener('click', function() {
        trainingForm.classList.remove('hidden');
        startTrainingBtn.disabled = true;
    });
    
    // Hide training form
    cancelTrainingBtn.addEventListener('click', function() {
        trainingForm.classList.add('hidden');
        startTrainingBtn.disabled = false;
    });
    
    // Handle new training form submission
    newTrainingForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(newTrainingForm);
        const trainingConfig = Object.fromEntries(formData.entries());
        
        // Convert numbers
        trainingConfig.max_rounds = parseInt(trainingConfig.max_rounds);
        trainingConfig.client_threshold = parseInt(trainingConfig.client_threshold);
        trainingConfig.learning_rate = parseFloat(trainingConfig.learning_rate);
        trainingConfig.step_size = parseInt(trainingConfig.step_size);
        trainingConfig.gamma = parseFloat(trainingConfig.gamma);
        
        // Send request to start training
        fetch('/training/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trainingConfig)
        })
        .then(response => response.json())
        .then(data => {
            alert('Training started successfully!');
            trainingForm.classList.add('hidden');
            startTrainingBtn.disabled = false;
            fetchTrainingStatus(); // Refresh data
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to start training. See console for details.');
        });
    });
    
    // Handle stop training button
    stopTrainingBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to stop the current training round?')) {
            fetch('/training/shutdown', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (data.success) {
                    
                    alert('Training stopped successfully!');
                    fetchTrainingStatus(); // Refresh data
                } else {
                    alert('Error stopping training: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Failed to stop training. See console for details.');
            });
        }
    });
    
    // Function to fetch current training status
    async function fetchTrainingStatus() {
        fetch('/view/current_round')
        .then(response => response.json())
        .then(data => {
            // Update training status section
            document.getElementById('currentStatus').textContent = data.is_training ? 'Active' : 'Inactive';
            document.getElementById('currentStatus').className = data.is_training ? 'status-active' : 'status-inactive';
            
            if (data.is_training) {
                document.getElementById('currentSuperRound').textContent = data.super_round;
                document.getElementById('currentRound').textContent = data.curr_round;
                document.getElementById('maxRounds').textContent = data.max_rounds;
                document.getElementById('currentLearningRate').textContent = data.learning_rate;
                document.getElementById('connectedClients').textContent = data.client_count || '0';
                document.getElementById('requiredClients').textContent = data.client_threshold;
            }
            
            // Update model performance section
            if (data.current_model) {
                document.getElementById('currentModelId').textContent = data.current_model.model_id;
                document.getElementById('currentAccuracy').textContent = 
                    data.current_model.accuracy !== null ? 
                    `${(data.current_model.accuracy * 100).toFixed(2)}%` : 'Not yet evaluated';
                // document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
                // document.getElementById('totalClientsTrained').textContent = data.trained_clients || '0';
            }
        })
        .catch(error => {
            console.error('Error fetching training status:', error);
        });
    }
    
    // Function to fetch connected clients
    async function fetchClients() {
        fetch('/api/clients')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('clientsTableBody');
            if (data.length > 0) {
                tableBody.innerHTML = '';
                data.forEach(client => {
                    const row = document.createElement('tr');
                    
                    // Determine status class
                    let statusClass = 'status-inactive';
                    if (client.state === 'ACTIVE') statusClass = 'status-active';
                    else if (client.state === 'TRAINING') statusClass = 'status-pending';
                    
                    row.innerHTML = `
                        <td>${client.client_id}</td>
                        <td><span class="status-indicator ${statusClass}"></span>${client.state}</td>
                        <td>${client.model_id || 'None'}</td>
                        <td>${client.has_trained ? 'Yes' : 'No'}</td>
                    `;
                    tableBody.appendChild(row);
                });
            } else {
                tableBody.innerHTML = '<tr><td colspan="4">No clients connected</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching clients:', error);
        });
    }
    
    // Function to fetch training history
    async function fetchTrainingHistory() {
        const superRound = parseInt(document.getElementById("currentSuperRound").textContent) || -1;
        fetch(`/view/history/${superRound}`)
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('trainingHistoryTableBody');
            if (data.length > 0) {
                tableBody.innerHTML = '';
                data.forEach(round => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${round.train_round}</td>
                        <td>${round.learning_rate}</td>
                        <td>${round.accuracy !== null ? (round.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                        <td>${round.num_clients || '0'}</td>
                    `;
                    tableBody.appendChild(row);
                });
            } else {
                tableBody.innerHTML = '<tr><td colspan="5">No training history available</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching training history:', error);
        });
    }

    async function fetchAndDisplayAccuracies() {
        const superRound = parseInt(document.getElementById("currentSuperRound").textContent) || -1;
        const trainRound = parseInt(document.getElementById("currentRound").textContent) || -1;
        
        if (superRound < 0 || trainRound < 0) return;
        
        fetch(`/view/models/${superRound}`)
        .then(response => response.json())
        .then(data => {
            const labels = data.map(entry => `${entry.round_id}`);
            const accuracies = data.map(entry => entry.accuracy);
            const modelIDs = data.map(entry => `${entry.model_id}`);
            console.log(`Accuracies: `, accuracies);
            const ctx = document.getElementById('accuracyChart').getContext('2d');

            // Calculate and update progress
            const maxRounds = parseInt(document.getElementById('maxRounds').textContent);
            let progress = maxRounds ? (modelIDs.length  / maxRounds * 100) : 0;
            document.getElementById('roundProgress').style.width = `${progress}%`;
            document.getElementById('roundProgress').textContent = `${Math.round(progress)}%`;

            document.getElementById('currentModelId').textContent = modelIDs[modelIDs.length - 1];
            document.getElementById('currentAccuracy').textContent = 
                accuracies[accuracies.length - 1] !== null ? 
                `${(accuracies[accuracies.length - 1] * 100).toFixed(2)}%` : 'Not yet evaluated';


            if (!accuracyChart) {
                accuracyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Model Accuracy Over Rounds',
                            data: accuracies,
                            borderColor: 'blue',
                            backgroundColor: 'rgba(0, 0, 255, 0.2)',
                            fill: false,
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                suggestedMax: 1
                            }
                        }
                    }
                });
            } else {
                accuracyChart.data.labels = labels;
                accuracyChart.data.datasets[0].data = accuracies;
                accuracyChart.update();
            }
            // Create a new chart
            
        })
        .catch(error => {
            console.error('Error fetching accuracy data: ', error);
        });
    }

    // Initial data fetch
    fetchTrainingStatus();
    // fetchClients();
    fetchTrainingHistory();
    fetchAndDisplayAccuracies();
    
    // Set up polling for continuous updates (every 5 seconds)
    setInterval(fetchTrainingStatus, 5000);
    setInterval(fetchTrainingHistory, 5000);
    setInterval(fetchAndDisplayAccuracies, 5000); // Less frequent for history

});