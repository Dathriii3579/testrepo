function submitfeedback() {
    // Get values inside the function
    const username = document.getElementById('name').value;
    const userage = document.getElementById('age').value;
    const useremail = document.getElementById('email').value;
    const userjob = document.getElementById('job').value;
    const userdesignation = document.getElementById('designation').value;
    const userproductchoice = document.getElementById('producttype').value;
    const feedback = document.getElementById('feedbacktext').value;

    // Display values in user info section
    document.getElementById('uname').innerText = username;
    document.getElementById('uage').innerText = userage;
    document.getElementById('uemail').innerText = useremail;
    document.getElementById('ujob').innerText = userjob;
    document.getElementById('udesignation').innerText = userdesignation;
    document.getElementById('uproductchoice').innerText = userproductchoice + " - ";
    document.getElementById('userfeedback').innerText = feedback;

    // Show the user info section
    document.getElementById('userinfo').style.display = 'block';

    // Show thank you message
    alert("Thank you for your valuable feedback!");
}

// Attach event to the submit button
document.getElementById('submitBtn').onclick = submitfeedback;

// Optional: allow Enter key to trigger submission
document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        submitfeedback();
    }
});
