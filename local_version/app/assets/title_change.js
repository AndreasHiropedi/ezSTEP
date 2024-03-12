if (!window.dash_clientside) { window.dash_clientside = {}; }
window.dash_clientside.clientside = {
    updateTitle: function(href) {
        // Logic to determine the title based on the URL
        let title = "ezSTEP";  // Default title
        let pathname = new URL(href).pathname;

        if (pathname.startsWith('/model-input/')) {
            let model_num = pathname.split('/').pop().slice(-1);
            // Check if model_num is not a number
            if (isNaN(model_num)) {
                title = "Invalid model number";
            } else {
                title = `Model ${model_num} input`;
            }
        } else if (pathname.startsWith('/model-output/')) {
            let model_num = pathname.split('/').pop().slice(-1);
            // Check if model_num is not a number
            if (isNaN(model_num)) {
                title = "Invalid model number";
            } else {
                title = `Model ${model_num} output`;
            }
        } else if (pathname.startsWith('/output-statistics/')) {
            title = "Testing output statistics"
        }

        document.title = title;
        return title;  // Return value is not used but required for callback
    }
};
