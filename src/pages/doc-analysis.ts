// src/pages/doc-analysis.ts
import { supabase } from '../supabaseClient';
import { marked } from 'marked';

let currentUserId: string | null = null;

const showLoading = (container: HTMLElement) => {
    container.innerHTML = '<div class="flex justify-center items-center"><div class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div></div>';
};

const handleFileUpload = async (file: File) => {
    if (!currentUserId) {
        alert('You must be logged in to upload a document.');
        return;
    }

    const outputContainer = document.getElementById('output-container') as HTMLElement;
    showLoading(outputContainer);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`http://localhost:8000/upload-document/${currentUserId}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Failed to upload document.');
        }

        const result = await response.json();
        outputContainer.innerHTML = `<p>${result.message}</p>`;

        // Now, ask for the analysis
        analyzeDocument(currentUserId, file.name);

    } catch (error) {
        console.error('Error uploading document:', error);
        outputContainer.innerHTML = `<p class="text-red-500">Error: ${(error as Error).message}</p>`;
    }
};

const analyzeDocument = async (userId: string, filename: string) => {
    const outputContainer = document.getElementById('output-container') as HTMLElement;
    outputContainer.innerHTML += '<p>Analyzing document...</p>';

    try {
        const response = await fetch(`http://localhost:8000/analyze-document/${userId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename }),
        });

        if (!response.ok) {
            throw new Error('Failed to analyze document.');
        }

        const result = await response.json();
        outputContainer.innerHTML = `
            <h3 class="text-lg font-bold mb-2">Analysis Result:</h3>
            <div class="prose">${marked(result.analysis)}</div>
        `;

    } catch (error) {
        console.error('Error analyzing document:', error);
        outputContainer.innerHTML += `<p class="text-red-500">Error: ${(error as Error).message}</p>`;
    }
}

const handleQuery = async (query: string) => {
    if (!currentUserId) {
        alert('You must be logged in to query a document.');
        return;
    }

    const queryOutputContainer = document.getElementById('query-output-container') as HTMLElement;
    showLoading(queryOutputContainer);

    try {
        const response = await fetch(`http://localhost:8000/query-document/${currentUserId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error('Failed to query document.');
        }

        const result = await response.json();
        queryOutputContainer.innerHTML = `
            <h3 class="text-lg font-bold mb-2">Query Answer:</h3>
            <div class="prose">${marked(result.answer)}</div>
        `;

    } catch (error) {
        console.error('Error querying document:', error);
        queryOutputContainer.innerHTML = `<p class="text-red-500">Error: ${(error as Error).message}</p>`;
    }
};


export const renderDocAnalysisPage = (container: HTMLElement) => {
    supabase.auth.getSession().then(({ data: { session } }) => {
        if (session) {
            currentUserId = session.user.id;
        }
    });

    container.innerHTML = `
        <div class="container mx-auto p-4">
            <h2 class="text-2xl font-bold mb-4">Document Analysis</h2>
            <div class="mb-4">
                <label for="document-upload" class="block text-lg font-medium mb-2">Upload Document</label>
                <input type="file" id="document-upload" class="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-violet-50 file:text-violet-700
                    hover:file:bg-violet-100
                "/>
            </div>
            <div id="output-container" class="mb-4"></div>
            <div class="mt-8">
                <h3 class="text-xl font-bold mb-4">Ask a question about the document</h3>
                <div class="flex gap-2">
                    <input type="text" id="query-input" class="border rounded-md p-2 w-full" placeholder="Enter your question...">
                    <button id="query-button" class="bg-blue-500 text-white px-4 py-2 rounded-md">Ask</button>
                </div>
                <div id="query-output-container" class="mt-4"></div>
            </div>
        </div>
    `;

    const uploadInput = document.getElementById('document-upload') as HTMLInputElement;
    uploadInput.addEventListener('change', (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (files && files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    const queryButton = document.getElementById('query-button') as HTMLButtonElement;
    const queryInput = document.getElementById('query-input') as HTMLInputElement;
    queryButton.addEventListener('click', () => {
        const query = queryInput.value;
        if (query) {
            handleQuery(query);
        }
    });
};
