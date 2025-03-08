document.addEventListener("DOMContentLoaded", function () {
    // Masquer la flèche au chargement de la page
    document.getElementById('arrowIcon').style.display = 'none';
});

// Configuration du graphique
const ctx = document.getElementById('priceChart').getContext('2d');
const priceChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [], // Étiquettes de temps
        datasets: [{
            label: 'Prix de clôture',
            data: [], // Données de prix
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
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
function updateChart(labels, data) {
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = data;
    priceChart.update();
}

// Variable pour stocker la connexion SSE
let eventSource = null;

// Gestion de la soumission du formulaire
document.getElementById('predictionForm').addEventListener('submit', function (event) {
    event.preventDefault();
    const symbol = document.getElementById('symbol').value;

    // Afficher l'indicateur de chargement
    document.getElementById('loadingIndicator').style.display = 'inline';

    // Fermer la connexion SSE existante
    if (eventSource) {
        eventSource.close();
    }

    // Ouvrir une nouvelle connexion SSE
    eventSource = new EventSource(`http://192.168.1.67:8000/updates/${symbol}`);

    // Gestion des messages reçus
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        // Masquer l'indicateur de chargement
        document.getElementById('loadingIndicator').style.display = 'none';

        // Convertir le changement et la probabilité en pourcentage
        const changePercentage = (data.change * 100).toFixed(2) + "%";
        const probabilityPercentage = (data.probability * 100).toFixed(2) + "%";

        // Mettre à jour le graphique et les résultats
        updateChart(data.historical_data.labels, data.historical_data.prices);
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
    };

    // Gestion des erreurs SSE
    eventSource.onerror = function(error) {
        console.error("Erreur SSE :", error);
        eventSource.close();
        document.getElementById('loadingIndicator').style.display = 'none';
        alert("La connexion au serveur a été perdue. Veuillez réessayer.");
    };
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
