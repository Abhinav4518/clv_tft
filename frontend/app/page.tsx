"use client";

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Area, ComposedChart, ResponsiveContainer } from 'recharts';
import { Activity, Search, AlertCircle, Calendar, Play } from 'lucide-react';

export default function Dashboard() {
  const [customerId, setCustomerId] = useState('C_001');
  const [targetDate, setTargetDate] = useState('2025-04-10');
  const [forecastData, setForecastData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Define the April 2025 Dataset Constraints
  // We lock maxDate to April 23 because the model needs 7 days of 'future' data (up to April 30)
  const minDate = "2025-04-01";
  const maxDate = "2025-04-23"; 

  const fetchForecast = async (overrideId?: string, overrideDate?: string) => {
    setLoading(true);
    setError('');
    setForecastData([]);
    
    const id = overrideId || customerId;
    const date = overrideDate || targetDate;

    try {
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const url = `${API_BASE}/predict/${id}?target_date=${date}`;
      
      const res = await fetch(url);
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'API connection failed.');
      }
      
      setForecastData(data.forecast);
    } catch (err: any) {
      setError(err.message || "Failed to fetch forecast. Is the FastAPI server running?");
    } finally {
      setLoading(false);
    }
  };

  // Helper to run a perfect sample instantly
  const runSample = () => {
    setCustomerId('C_001');
    setTargetDate('2025-04-10');
    fetchForecast('C_001', '2025-04-10');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 md:p-16 font-sans">
      <div className="max-w-5xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-10">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Activity className="text-blue-500 w-8 h-8" />
              <h1 className="text-3xl font-bold text-white">CLV Forecaster</h1>
            </div>
            <p className="text-slate-400">Temporal Fusion Transformer (TFT) Engine • April 2025 Simulation</p>
          </div>
          
          <button 
            onClick={runSample}
            className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-blue-400 border border-blue-900/30 px-4 py-2 rounded-lg text-sm font-medium transition-all"
          >
            <Play className="w-4 h-4 fill-current" />
            Try Sample Scenario
          </button>
        </div>

        {/* Input & Controls */}
        <div className="flex flex-col md:flex-row gap-4 mb-10 bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg">
          
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 text-slate-500 w-5 h-5" />
            <input 
              type="text" 
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value.toUpperCase())}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg py-2.5 pl-10 pr-4 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
              placeholder="Customer ID (e.g., C_001)"
            />
          </div>

          <div className="flex-1 relative">
            <Calendar className="absolute left-3 top-3 text-slate-500 w-5 h-5" />
            <input 
              type="date" 
              value={targetDate}
              onChange={(e) => setTargetDate(e.target.value)}
              min={minDate}
              max={maxDate}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg py-2.5 pl-10 pr-4 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all [color-scheme:dark]"
            />
          </div>

          <button 
            onClick={() => fetchForecast()}
            disabled={loading || !customerId}
            className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 px-8 py-2.5 rounded-lg font-semibold transition-colors flex items-center justify-center min-w-[160px]"
          >
            {loading ? 'Running AI...' : 'Run Forecast'}
          </button>
        </div>

        {/* Error Handling */}
        {error && (
          <div className="flex items-center gap-2 bg-red-950/50 border border-red-900 text-red-400 px-4 py-3 rounded-lg mb-8">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        )}

        {/* The Forecast Visualization */}
        {forecastData.length > 0 ? (
          <div className="bg-slate-900 p-6 md:p-8 rounded-xl border border-slate-800 shadow-2xl">
            <div className="mb-8">
              <h2 className="text-xl font-semibold text-white">7-Day Spend Projection</h2>
              <p className="text-sm text-slate-400">Showing the P50 median with 80% confidence interval bounds.</p>
            </div>
            
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={forecastData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                  
                  <XAxis 
                    dataKey="date" 
                    stroke="#64748B" 
                    axisLine={false}
                    tickLine={false}
                    dy={10}
                  />
                  
                  <YAxis 
                    stroke="#64748B" 
                    tickFormatter={(val) => `$${val}`}
                    axisLine={false}
                    tickLine={false}
                    dx={-10}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #1E293B', borderRadius: '8px', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                    formatter={(value: any) => [value ? `$${Number(value).toFixed(2)}` : '$0.00', 'Spend']}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Legend verticalAlign="top" height={36} iconType="circle" />
                  
                  <Area 
                    type="monotone" 
                    dataKey="p90_upper_bound" 
                    stroke="none" 
                    fill="#3B82F6" 
                    fillOpacity={0.15} 
                    name="80% Confidence Bound" 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p10_lower_bound" 
                    stroke="none" 
                    fill="#0F172A" 
                    fillOpacity={1} 
                    name="" 
                    legendType="none"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="p50_median" 
                    stroke="#3B82F6" 
                    strokeWidth={3} 
                    dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, strokeWidth: 0 }}
                    name="Median Forecast (P50)" 
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : !loading && (
          <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-slate-800 rounded-xl">
             <Calendar className="w-12 h-12 text-slate-700 mb-4" />
             <p className="text-slate-500">Select a date in April 2025 to generate a forecast.</p>
          </div>
        )}
      </div>
    </div>
  );
}
