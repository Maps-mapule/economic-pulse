function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.style.display = 'none';
    });
    document.getElementById(screenId).style.display = 'block';
}

// Example: Show landing page after sign in
document.getElementById('signin-button').addEventListener('click', function() {
    showScreen('landing-page');
});

// Example: Show forgot password screen
document.getElementById('forgot-password-link').addEventListener('click', function() {
    showScreen('forgot-password-screen');
});

// Example: Show sign in screen from forgot password screen
document.getElementById('back-to-signin-link').addEventListener('click', function() {
    showScreen('signin-screen');
});

// Show Client Profiles page
document.getElementById('client-profile-button').addEventListener('click', function() {
    showScreen('client-profile-page');
});

// Show Client Info page
document.getElementById('client-info-button').addEventListener('click', function() {
    showScreen('client-info-page');
});

// Show Prediction Process page
document.getElementById('prediction-process-button').addEventListener('click', function() {
    showScreen('prediction-process-page');
});

// Show Offer Loan to Client page
document.getElementById('offer-loan-button').addEventListener('click', function() {
    showScreen('offer-loan-page');
});

// Back to Landing page buttons
document.getElementById('back-to-landing-button1').addEventListener('click', function() {
    showScreen('landing-page');
});
document.getElementById('back-to-landing-button2').addEventListener('click', function() {
    showScreen('landing-page');
});
document.getElementById('back-to-landing-button3').addEventListener('click', function() {
    showScreen('landing-page');
});
document.getElementById('back-to-landing-button4').addEventListener('click', function() {
    showScreen('landing-page');
});