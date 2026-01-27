'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

type Citation = {
    document_title: string;
    page_no: number;
    section: string;
    url: string;
};

type Message = {
    role: 'user' | 'assistant';
    content: string;
    citations?: Citation[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE!;

export default function Page() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [user, setUser] = useState<any>(null);
    const [accessToken, setAccessToken] = useState<string | null>(null);

    const [pdfUrl, setPdfUrl] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [question, setQuestion] = useState('');
    const [ingested, setIngested] = useState(false);
    const [loading, setLoading] = useState(false);

    /* ------------------ AUTH ------------------ */
    useEffect(() => {
        supabase.auth.getSession().then(({ data }) => {
            setUser(data.session?.user ?? null);
            setAccessToken(data.session?.access_token ?? null);
        });

        const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user ?? null);
            setAccessToken(session?.access_token ?? null);
        });

        return () => listener.subscription.unsubscribe();
    }, []);

    async function signUp() {
        if (!email || !password) return alert('Enter email and password');
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) return alert(error.message);
        alert('Signup successful! Please check your email for confirmation.');
    }

    async function signIn() {
        if (!email || !password) return alert('Enter email and password');
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) return alert(error.message);
    }

    async function signOut() {
        await supabase.auth.signOut();
        setUser(null);
        setAccessToken(null);
    }

    /* ------------------ INGEST ------------------ */
    async function ingestPdf() {
        if (!pdfUrl || !accessToken) return;

        await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${accessToken}`,
            },
            body: JSON.stringify({ filepath: pdfUrl }),
        });
        console.log("Access token:", accessToken);


        setIngested(true);
        alert('PDF ingested successfully');
    }

    /* ------------------ CHAT ------------------ */
    async function askQuestion() {
        if (!question || !accessToken) return;

        const newMessages = [...messages, { role: 'user' as const, content: question }];
        setMessages(newMessages);
        setQuestion('');
        setLoading(true);

        const res = await fetch(`${API_BASE}/v1/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${accessToken}`,
            },
            body: JSON.stringify({ messages: newMessages }),
        });

        const data = await res.json();

        setMessages([
            ...newMessages,
            { ...data.choices[0].message, citations: data.citations || [] },
        ]);

        setLoading(false);
    }

    const isAuthed = Boolean(user);

    return (
        <main className="min-h-screen bg-gray-100 flex justify-center p-6">
            <div className="w-full max-w-3xl bg-white rounded-xl shadow p-6 flex flex-col">
                <h1 className="text-2xl font-bold mb-6">📄 PDF RAG Chat</h1>

                {/* ------------------ AUTH FORM ------------------ */}
                {!isAuthed && (
                    <div className="mb-4 flex flex-col gap-2">
                        <input
                            type="email"
                            placeholder="Email"
                            className="border rounded px-3 py-2"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                        <input
                            type="password"
                            placeholder="Password"
                            className="border rounded px-3 py-2"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={signUp}
                                className="bg-blue-600 text-white px-4 py-2 rounded"
                            >
                                Sign Up
                            </button>
                            <button
                                onClick={signIn}
                                className="bg-green-600 text-white px-4 py-2 rounded"
                            >
                                Sign In
                            </button>
                        </div>
                    </div>
                )}

                {isAuthed && (
                    <div className="mb-4">
                        <p className="text-sm text-gray-700 mb-2">Logged in as {user.email}</p>
                        <button
                            onClick={signOut}
                            className="bg-red-600 text-white px-4 py-2 rounded"
                        >
                            Logout
                        </button>
                    </div>
                )}

                {/* ------------------ PDF Upload ------------------ */}
                <div className="flex gap-2 mb-6">
                    <input
                        className="flex-1 border rounded px-3 py-2"
                        placeholder="Paste PDF URL"
                        value={pdfUrl}
                        onChange={(e) => setPdfUrl(e.target.value)}
                        disabled={!isAuthed}
                    />
                    <button
                        onClick={ingestPdf}
                        disabled={!isAuthed}
                        className="bg-blue-600 text-white px-4 rounded disabled:opacity-50"
                    >
                        Ingest
                    </button>
                </div>

                {/* ------------------ Chat ------------------ */}
                <div className="flex-1 border rounded p-4 overflow-y-auto space-y-3 mb-4">
                    {!ingested && (
                        <div className="text-sm text-gray-500">
                            Ingest a PDF to start chatting
                        </div>
                    )}

                    {messages.map((m, i) => (
                        <div
                            key={i}
                            className={`max-w-[80%] p-3 rounded space-y-2 ${m.role === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-200'
                                }`}
                        >
                            <div>{m.content}</div>
                            {m.role === 'assistant' && (m.citations?.length ?? 0) > 0 && (
                                <div className="text-xs text-gray-600 border-t pt-2 space-y-1">
                                    <div className="font-semibold">Sources:</div>
                                    {m.citations?.map((c, idx) => (
                                        <div key={idx}>
                                            📄 <span className="font-medium">{c.document_title}</span>
                                            {c.page_no && <> — page {c.page_no}</>}
                                            {c.section && <> — <em>{c.section}</em></>}
                                            {' '}
                                            <a
                                                href={c.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-blue-600 underline"
                                            >
                                                view
                                            </a>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}

                    {loading && (
                        <div className="text-sm text-gray-500">Thinking...</div>
                    )}
                </div>

                {/* ------------------ Chat Input ------------------ */}
                <div className="flex gap-2">
                    <input
                        className="flex-1 border rounded px-3 py-2"
                        placeholder="Ask a question about the PDF..."
                        value={question}
                        disabled={!ingested || !isAuthed}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
                    />
                    <button
                        onClick={askQuestion}
                        disabled={!ingested || !isAuthed}
                        className="bg-green-600 text-white px-4 rounded disabled:opacity-50"
                    >
                        Ask
                    </button>
                </div>
            </div>
        </main>
    );
}
