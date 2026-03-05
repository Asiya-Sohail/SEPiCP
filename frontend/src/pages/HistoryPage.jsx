import React, { useState, useEffect } from 'react';
import {
  History as HistoryIcon,
  Search,
  Filter,
  Download,
  Eye,
  FileSpreadsheet,
  RefreshCw,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { Card } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { useNavigate } from 'react-router-dom';

const HistoryPage = () => {
  const [datasets, setDatasets] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const token = localStorage.getItem('research_token');
      const response = await fetch("http://127.0.0.1:8000/api/datasets/", {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setDatasets(data);
      } else if (response.status === 401) {
        alert("Session Expired. Please log out and back in.");
      }
    } catch (err) {
      console.error("Failed to fetch dataset history", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-12 animate-in fade-in duration-700 pb-20">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <h1 className="text-5xl font-black tracking-tighter text-slate-900 leading-tight">Institutional Audit Log</h1>
          <p className="text-slate-500 mt-3 text-xl font-bold">Comprehensive history of previously cleaned datasets.</p>
        </div>
        <div className="flex gap-4">
          <Button onClick={fetchHistory} variant="outline" className="h-14 px-8 rounded-2xl font-black border-slate-200 bg-white/50 backdrop-blur-sm text-[13px] uppercase tracking-widest shadow-sm hover:bg-white hover:border-[#1e3a8a] transition-all">
            <RefreshCw className={cn("w-5 h-5 mr-3", isLoading && "animate-spin")} />
            Sync Database
          </Button>
        </div>
      </div>

      <Card className="rounded-[40px] border-slate-200/60 shadow-2xl shadow-slate-200/40 overflow-hidden bg-white/80 backdrop-blur-sm border-b-4 border-b-[#1e3a8a]">
        <div className="p-10 border-b border-slate-100 flex flex-wrap gap-6 justify-between items-center bg-slate-50/30">
          <div className="relative flex-1 max-w-xl">
            <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 z-10" />
            <Input
              type="text"
              placeholder="Search dataset records..."
              className="pl-14 h-16 bg-white border-slate-200 rounded-2xl font-bold text-slate-800 focus:ring-4 focus:ring-blue-100 border-none shadow-inner"
            />
          </div>
          <div className="flex items-center gap-4">
            <Badge className="bg-emerald-50 text-emerald-700 py-2 px-5 rounded-full border border-emerald-100 font-black text-[11px] uppercase tracking-widest">
              Live Database
            </Badge>
          </div>
        </div>

        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="bg-slate-50/50 border-b border-slate-100 hover:bg-slate-50/50">
                <TableHead className="px-10 py-6 text-[11px] font-black text-slate-400 uppercase tracking-[0.2em]">Dataset Type</TableHead>
                <TableHead className="px-10 py-6 text-[11px] font-black text-slate-400 uppercase tracking-[0.2em]">Rows</TableHead>
                <TableHead className="px-10 py-6 text-[11px] font-black text-slate-400 uppercase tracking-[0.2em]">Upload Timestamp</TableHead>
                <TableHead className="px-10 py-6 text-[11px] font-black text-slate-400 uppercase tracking-[0.2em] text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={4} className="h-40 text-center text-slate-400 font-medium">Fetching history from server...</TableCell>
                </TableRow>
              ) : datasets.length > 0 ? (
                datasets.map((item) => (
                  <TableRow key={item.id} className="hover:bg-blue-50/50 transition-colors group border-b border-slate-50 last:border-0">
                    <TableCell className="px-10 py-6">
                      <div className="flex items-center gap-5">
                        <div className="w-14 h-14 rounded-[20px] flex items-center justify-center shadow-inner transition-transform group-hover:scale-110 bg-amber-50 text-amber-600">
                          <FileSpreadsheet className="w-6 h-6" />
                        </div>
                        <span className="text-[13px] font-black uppercase tracking-widest text-[#1e3a8a]">XLSX UPLOAD</span>
                      </div>
                    </TableCell>
                    <TableCell className="px-10 py-6">
                      <p className="text-sm font-black text-slate-800 leading-relaxed">{item.cleaned_data?.length || 0} Filtered Rows</p>
                    </TableCell>
                    <TableCell className="px-10 py-6 text-sm text-slate-500 font-bold">
                      {new Date(item.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell className="px-10 py-6 text-right">
                      <Button
                        onClick={() => {
                          localStorage.setItem('load_dataset_id', item.id);
                          navigate('/upload');
                        }}
                        variant="ghost"
                        className="rounded-2xl text-slate-400 hover:text-[#1e3a8a] text-xs font-bold uppercase hover:bg-white shadow-sm border border-transparent hover:border-slate-100 transition-all"
                      >
                        Load to Workstation
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={4} className="h-40 text-center text-slate-400 font-medium">
                    No datasets found in the database. Upload a file to populate history.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        <div className="p-6 border-t border-slate-100 bg-slate-50/30 flex items-center justify-between">
          <p className="text-sm text-slate-500 font-medium">Showing <span className="text-slate-900 font-bold">{datasets.length}</span> datasets configured</p>
        </div>
      </Card>
    </div>
  );
};

export default HistoryPage;

