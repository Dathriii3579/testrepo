let count = 0;

function increaseCount() {
    count++;
    displayCount();
    checkCountValue();
}

function displayCount() {
    document.getElementById('countDisplay').innerHTML = count;
}

function checkCountValue() {
    if (count === 10) {
        alert("Your Instagram has gained 10 followers! 🎉 Keep going!");
    } else if (count === 20) {
        alert("Your post has gained 20 followers! 🚀 Keep it up!");
    }
}
