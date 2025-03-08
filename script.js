document.addEventListener("DOMContentLoaded", function () {
    // Masquer la flèche au chargement de la page
    document.getElementById('arrowIcon').style.display = 'none';
});

// Configuration du graphique des prédictions
const predictionCtx = document.getElementById('predictionChart').getContext('2d');
const predictionChart = new Chart(predictionCtx, {
    type: 'line',
    data: {
        labels: [], // Étiquettes de temps
        datasets: [
            {
                label: 'Prix prédit',
                data: [], // Données de prix prédit
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
                label: 'Prix actuel',
                data: [], // Données de prix actuel
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'Temps' } },
            y: { title: { display: true, text: 'Prix' } }
        }
    }
});

// Fonction pour mettre à jour le graphique
function updateChart(labels, predictedPrices, actualPrices) {
    // Vérifier que les données sont valides
    if (!Array.isArray(predictedPrices) || !Array.isArray(actualPrices)) {
        console.error("Données invalides pour la mise à jour du graphique :", { predictedPrices, actualPrices });
        return;
    }

    // Mettre à jour les données du graphique
    predictionChart.data.labels = labels;
    predictionChart.data.datasets[0].data = predictedPrices;
    predictionChart.data.datasets[1].data = actualPrices;
    predictionChart.update();
}

// Fonction pour récupérer les données de prédiction
function fetchPredictionData(symbol) {
    // Afficher l'indicateur de chargement
    document.getElementById('loadingIndicator').style.display = 'inline';

    // Masquer l'indicateur de chargement après un délai (par exemple, 60 secondes)
    const loadingTimeout = setTimeout(function () {
        document.getElementById('loadingIndicator').style.display = 'none';
        alert("Le chargement a pris trop de temps. Veuillez réessayer.");
    }, 60000); // 60 secondes

    // Récupérer les données de prédiction
    fetch(`http://192.168.1.67:8000/updates/${symbol}`)
        .then(response => response.json())
        .then(data => {
            // Masquer l'indicateur de chargement
            document.getElementById('loadingIndicator').style.display = 'none';
            clearTimeout(loadingTimeout); // Annuler le timeout

            // Convertir le changement et la probabilité en pourcentage
            const changePercentage = (data.change * 100).toFixed(2) + "%";
            const probabilityPercentage = (data.probability * 100).toFixed(2) + "%";

            // Mettre à jour le graphique et les résultats
            updateChart(data.historical_data.labels, data.historical_data.predicted_prices, data.historical_data.actual_prices);
            document.getElementById('symbolResult').textContent = data.symbol;
            document.getElementById('predictedPriceResult').textContent = data.predicted_price.toFixed(2);
            document.getElementById('actualPriceResult').textContent = data.actual_price.toFixed(2);
            document.getElementById('changeResult').textContent = changePercentage;
            document.getElementById('probabilityResult').textContent = probabilityPercentage;
            document.getElementById('actionResult').textContent = data.action;

            // Mettre à jour la flèche
            updateArrow(data.action, data.probability, data.change);

            // Afficher une notification si l'action est "Acheter" ou "Vendre"
            if (data.action !== "Attendre") {
                showNotification(data.action);
            }
        })
        .catch(error => {
            console.error("Erreur lors de la récupération des données :", error);
            document.getElementById('loadingIndicator').style.display = 'none';
            clearTimeout(loadingTimeout); // Annuler le timeout
            alert("Erreur lors de la récupération des données. Veuillez réessayer.");
        });
}

// Gestion de la soumission du formulaire
document.getElementById('predictionForm').addEventListener('submit', function (event) {
    event.preventDefault();
    const symbol = document.getElementById('symbol').value;

    // Récupérer les données immédiatement
    fetchPredictionData(symbol);

    // Mettre à jour les données toutes les 15 minutes
    setInterval(() => {
        fetchPredictionData(symbol);
    }, 15 * 60 * 1000); // 15 minutes en millisecondes
});

// Fonction pour mettre à jour la flèche
function updateArrow(action, probability, change) {
    const arrowIcon = document.getElementById('arrowIcon');

    // Afficher la flèche s'il y a des données
    arrowIcon.style.display = 'inline';

    // Déterminer la direction de la flèche en fonction du changement
    if (change >= 0) {
        arrowIcon.className = "fas fa-arrow-up"; // Flèche vers le haut pour une tendance haussière
        // Changer la couleur en nuances de vert
        if (probability >= 0.7) {
            arrowIcon.style.color = "darkgreen"; // Vert foncé pour probabilité >= 70 %
        } else if (probability >= 0.5) {
            arrowIcon.style.color = "limegreen"; // Vert clair pour probabilité entre 50 % et 70 %
        } else {
            arrowIcon.style.color = "lightgreen"; // Vert très clair pour probabilité < 50 %
        }
    } else {
        arrowIcon.className = "fas fa-arrow-down"; // Flèche vers le bas pour une tendance baissière
        // Changer la couleur en nuances de rouge
        if (probability >= 0.7) {
            arrowIcon.style.color = "darkred"; // Rouge foncé pour probabilité >= 70 %
        } else if (probability >= 0.5) {
            arrowIcon.style.color = "orange"; // Orange pour probabilité entre 50 % et 70 %
        } else {
            arrowIcon.style.color = "lightcoral"; // Rouge clair pour probabilité < 50 %
        }
    }
}

// Fonction pour afficher des notifications
function showNotification(message) {
    if (Notification.permission === "granted") {
        new Notification("Nouvelle opportunité", { body: message });
    } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then(permission => {
            if (permission === "granted") {
                new Notification("Nouvelle opportunité", { body: message });
            }
        });
    }
}