if (!window.dash_clientside) { window.dash_clientside = {}; }
window.dash_clientside.clientside = {
    updateTitle: function(href) {
        // Logic to determine the title based on the URL
        let title = "BioNetTrain";  // Default title
        let pathname = new URL(href).pathname;

        if (pathname.startsWith('/model-input/')) {
            let model_num = pathname.split('/').pop().slice(-1);
            title = `Model ${model_num} input`;
        } else if (pathname.startsWith('/model-output/')) {
            let model_num = pathname.split('/').pop().slice(-1);
            title = `Model ${model_num} output`;
        } else if (pathname.startsWith('/output-statistics/')) {
            if (pathname.includes('RMSE')) {
                title = 'RMSE plot';
            } else if (pathname.includes('R-squared')) {
                title = 'R-squared plot';
            } else if (pathname.includes('MAE')) {
                title = 'MAE plot';
            } else if (pathname.includes('Percentage within 2-fold error')) {
                title = 'Percentage within 2-fold error plot';
            }
        }

        document.title = title;
        return title;  // Return value is not used but required for callback
    }
};
