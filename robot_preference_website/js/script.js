document.addEventListener('DOMContentLoaded', function() {
    // Initialize Supabase client
    const supabaseUrl = 'https://tnmtwecxuybianbksciw.supabase.co';
    const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRubXR3ZWN4dXliaWFuYmtzY2l3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc5NTg5OTAsImV4cCI6MjA2MzUzNDk5MH0.StW7kfEdhBTbX9SvrnfcKBor_oDxVEkQzV-w1bgzD-E';
    const supabase = supabase.createClient(supabaseUrl, supabaseKey);
    
    // Sample data - in production, this would come from your server
    const trajectoryPairs = [
        {
            id: 1,
            trajectoryA: {
                id: 'traj_1a',
                videoSrc: 'assets/traj_1a.mp4',
            },
            trajectoryB: {
                id: 'traj_1b',
                videoSrc: 'assets/traj_1b.mp4',
            }
        },
        {
            id: 2,
            trajectoryA: {
                id: 'traj_2a',
                videoSrc: 'assets/traj_2a.mp4',
            },
            trajectoryB: {
                id: 'traj_2b',
                videoSrc: 'assets/traj_2b.mp4',
            }
        },
        // Add more pairs as needed
    ];
    
    // Elements
    const trajectoryAVideo = document.getElementById('trajectory-A');
    const trajectoryBVideo = document.getElementById('trajectory-B');
    const preferAButton = document.getElementById('prefer-A');
    const preferEqualButton = document.getElementById('prefer-equal');
    const preferBButton = document.getElementById('prefer-B');
    const prevPairButton = document.getElementById('prev-pair');
    const nextPairButton = document.getElementById('next-pair');
    const submitAllButton = document.getElementById('submit-all');
    const currentPairElement = document.getElementById('current-pair');
    const totalPairsElement = document.getElementById('total-pairs');
    const progressElement = document.getElementById('progress');
    const thanksMessage = document.getElementById('thanks-message');
    const restartButton = document.getElementById('restart-button');
    
    // State
    let currentPairIndex = 0;
    const preferences = [];
    
    // Initialize
    function init() {
        // Set total pairs count
        totalPairsElement.textContent = trajectoryPairs.length;
        
        // Load first pair
        loadTrajectoryPair(currentPairIndex);
        
        // Update UI
        updateUI();
    }
    
    // Load trajectory pair videos
    function loadTrajectoryPair(index) {
        const pair = trajectoryPairs[index];
        
        // Set video sources
        trajectoryAVideo.src = pair.trajectoryA.videoSrc;
        trajectoryBVideo.src = pair.trajectoryB.videoSrc;
        
        // Load the videos
        trajectoryAVideo.load();
        trajectoryBVideo.load();
        
        // Update current pair number display
        currentPairElement.textContent = index + 1;
        
        // Update progress bar
        const progress = ((index + 1) / trajectoryPairs.length) * 100;
        progressElement.style.width = `${progress}%`;
    }
    
    // Record user preference
    function recordPreference(preference) {
        const pair = trajectoryPairs[currentPairIndex];
        
        preferences[currentPairIndex] = {
            pairId: pair.id,
            trajectoryA: pair.trajectoryA.id,
            trajectoryB: pair.trajectoryB.id,
            preference: preference,
            timestamp: new Date().toISOString()
        };
        
        // Enable next pair button
        nextPairButton.disabled = false;
        
        // If all pairs have preferences, enable submit button
        if (preferences.length === trajectoryPairs.length) {
            submitAllButton.disabled = false;
        }
    }
    
    // Go to next pair
    function goToNextPair() {
        if (currentPairIndex < trajectoryPairs.length - 1) {
            currentPairIndex++;
            loadTrajectoryPair(currentPairIndex);
            updateUI();
        }
    }
    
    // Go to previous pair
    function goToPrevPair() {
        if (currentPairIndex > 0) {
            currentPairIndex--;
            loadTrajectoryPair(currentPairIndex);
            updateUI();
        }
    }
    
    // Update UI based on current state
    function updateUI() {
        // Update navigation buttons
        prevPairButton.disabled = currentPairIndex === 0;
        nextPairButton.disabled = !preferences[currentPairIndex];
        
        // Check if we should enable submit button
        const allPairsRated = preferences.length === trajectoryPairs.length && 
                            !preferences.includes(undefined);
        submitAllButton.disabled = !allPairsRated;
    }
    
    // Submit all preferences
    async function submitAllPreferences() {
        console.log('Submitting preferences:', preferences);
        
        // Create a submission object with metadata
        const submission = {
            preferences: preferences,
            submitted_at: new Date().toISOString(),
            user_agent: navigator.userAgent,
            // Add any other metadata you want to track
        };
        
        try {
            // Insert preferences into Supabase
            const { data, error } = await supabase
                .from('robot_preferences')
                .insert([submission]);
                
            if (error) {
                console.error('Error saving preferences:', error);
                alert('There was an error saving your preferences. Please try again.');
            } else {
                console.log('Preferences saved successfully:', data);
                thanksMessage.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Exception when saving preferences:', error);
            alert('An unexpected error occurred. Please try again.');
        }
    }
    
    // Restart the process
    function restart() {
        currentPairIndex = 0;
        preferences.length = 0;
        loadTrajectoryPair(currentPairIndex);
        thanksMessage.classList.add('hidden');
        updateUI();
    }
    
    // Event listeners
    preferAButton.addEventListener('click', function() {
        recordPreference('A');
        updateUI();
    });
    
    preferEqualButton.addEventListener('click', function() {
        recordPreference('equal');
        updateUI();
    });
    
    preferBButton.addEventListener('click', function() {
        recordPreference('B');
        updateUI();
    });
    
    prevPairButton.addEventListener('click', goToPrevPair);
    nextPairButton.addEventListener('click', goToNextPair);
    submitAllButton.addEventListener('click', submitAllPreferences);
    restartButton.addEventListener('click', restart);
    
    // Initialize the page
    init();
}); 