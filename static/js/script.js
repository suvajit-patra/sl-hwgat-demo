const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const countdownDiv = document.getElementById('countdown');
const recordBtn = document.getElementById('record-btn');
const dataList = document.getElementById('result-list');
const processingLogo = document.getElementById('processing-window');
const topKSlider = document.getElementById('top_k');
const topKText = document.getElementById('top_k_value');
const recordDurationSlider = document.getElementById('record_duration');
const recordDurationText = document.getElementById('record_duration_value');

var is_processing = false

topKText.innerHTML = topKSlider.value; // Display the default slider value
recordDurationText.innerHTML = recordDurationSlider.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
topKSlider.oninput = function() {
    topKText.innerHTML = this.value;
}

// Update the current slider value (each time you drag the slider handle)
recordDurationSlider.oninput = function() {
    recordDurationText.innerHTML = this.value;
}

// Access the webcam
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        drawToCanvas();
    } catch (error) {
        console.error('Error accessing the webcam: ', error);
    }
}

function showAlert(msg) {
    alert(msg);
}

function toggleCitationBox(id) {
    citebox = document.getElementById(id);
    if (citebox.style.display == 'block'){
        citebox.style.display = 'none';
    }
    else {
        citebox.style.display = 'block';
    }
}

// Draw video frames to canvas and send frames to API on click
function drawToCanvas() {
    recordBtn.addEventListener('click', () => {
        console.log('rec click')
        if (!is_processing) {
            startRecording(3, parseInt(recordDurationSlider.value)); // Start countdown for 3 seconds
        } else {
            alert('Processing video')
        }
    });

    function startRecording(wait_time, record_duration) {
        let countdown = record_duration + wait_time;
        countdownDiv.style.color = 'red';
        countdownDiv.style.backgroundColor = 'white';
        countdownDiv.style.opacity = '0.5';
        countdownDiv.style.fontSize = '100px';
        countdownDiv.textContent = wait_time; // Display countdown
        const countdownInterval = setInterval(async () => {
            countdown -= 1;
            wait_time -= 1;
            if (wait_time > 0) {
                countdownDiv.textContent = wait_time;
            } else {
                if (countdown == record_duration) {
                    navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                    const recorder = new MediaRecorder(stream);

                    recorder.ondataavailable = (event) => {
                        const blob = new Blob([event.data], { type: 'video/webm' });
                        is_processing = true;
                        processingLogo.style.display = 'block';
                        const url = URL.createObjectURL(blob);

                        // Send the video URL to the Flask API
                        fetch('/record', {
                            method: 'POST',
                            body: blob
                        })
                        .then(response => {
                            console.log('Video sent successfully:', response);
                        })
                        .catch(error => {
                            console.error('Error sending video:', error);
                        });
                    };

                    recorder.start();

                    // Stop recording after a certain time (adjust as needed)
                    setTimeout(() => {
                        recorder.stop();
                    }, record_duration*1000); // 10 seconds
                    })
                    .catch(error => {
                    console.error('Error accessing camera:', error);
                    });
                }
                countdownDiv.style.color = 'green';
                countdownDiv.style.backgroundColor = 'transparent';
                countdownDiv.style.opacity = '1';
                countdownDiv.style.fontSize = '40px';
                countdownDiv.textContent = countdown;
            }

            if (countdown <= 0) {
                clearInterval(countdownInterval);
                countdownDiv.textContent = ''; // Clear countdown
                countdownDiv.style.opacity = '0';
                
                await fetch('/result')
                    .then(response => response.json())
                    .then(data => {
                        is_processing = false;
                        processingLogo.style.display = 'none';
                        dataList.innerHTML = '';

                        data.forEach(item => {
                            const listItem = document.createElement('p');
                            listItem.textContent = item;
                            dataList.appendChild(listItem);
                        });
                    })
                    .catch(error => console.error('Error:', error));
            }

            // Capture the frame from the canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
        }, 1000); // Update every second
    }

    function captureFrame() {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    function render() {
        captureFrame();
        requestAnimationFrame(render);
    }

    render(); // Start rendering loop
}

// Start the webcam when the page loads
window.onload = startWebcam;