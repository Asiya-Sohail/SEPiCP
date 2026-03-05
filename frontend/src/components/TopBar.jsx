import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { HelpCircle, Bell, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';

const TopBar = () => {
    const location = useLocation();
    const pathnames = location.pathname.split('/').filter((x) => x);

    const getBreadcrumbs = () => {
        const crumbs = [{ label: 'Home', path: '/' }];

        if (pathnames.includes('upload')) {
            crumbs.push({ label: 'Projects', path: '/' });
            crumbs.push({ label: 'Alpha Study', path: '/' });
            crumbs.push({ label: 'Upload Data', path: '/upload' });
        } else if (pathnames.includes('analysis')) {
            crumbs.push({ label: 'Projects', path: '/' });
            crumbs.push({ label: 'Alpha Study', path: '/' });
            crumbs.push({ label: 'Configuration', path: '/analysis' });
        } else if (pathnames.includes('results')) {
            crumbs.push({ label: 'Projects', path: '/' });
            crumbs.push({ label: 'Alpha Study', path: '/' });
            crumbs.push({ label: 'Results Report', path: '/results' });
        } else if (pathnames.length === 0) {
            // Dashboard
        } else {
            pathnames.forEach((value, index) => {
                const last = index === pathnames.length - 1;
                const to = `/${pathnames.slice(0, index + 1).join('/')}`;
                crumbs.push({ label: value.charAt(0).toUpperCase() + value.slice(1), path: to });
            });
        }
        return crumbs;
    };

    const breadcrumbs = getBreadcrumbs();

    return (
        <header className="h-14 flex items-center justify-between px-6 lg:px-10 border-b border-slate-100 bg-white/50 backdrop-blur-md sticky top-0 z-40">
            <nav className="flex items-center gap-2 text-[12px] font-medium transition-all ml-12 lg:ml-0">
                {breadcrumbs.map((crumb, index) => (
                    <React.Fragment key={`${crumb.path}-${index}`}>
                        {index > 0 && <ChevronRight className="w-3 h-3 text-slate-300 mx-0.5" />}
                        <Link
                            to={crumb.path}
                            className={cn(
                                "transition-colors truncate max-w-[80px] sm:max-w-none",
                                index === breadcrumbs.length - 1
                                    ? "text-slate-900 font-bold"
                                    : "text-slate-400 hover:text-slate-600"
                            )}
                        >
                            {crumb.label}
                        </Link>
                    </React.Fragment>
                ))}
            </nav>

            <div className="flex items-center gap-4 sm:gap-6">
                <button className="flex items-center gap-2 text-[12px] font-bold text-slate-500 hover:text-[#1e3a8a] transition-all group">
                    <HelpCircle className="w-4 h-4 text-slate-400 group-hover:text-[#1e3a8a]" />
                    <span className="hidden sm:inline">Help Center</span>
                </button>
                <button className="relative p-1.5 rounded-lg text-slate-400 hover:text-[#1e3a8a] hover:bg-slate-50 transition-all group">
                    <Bell className="w-4.5 h-4.5" />
                    <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-red-500 border-2 border-white rounded-full"></span>
                </button>
            </div>
        </header>
    );
};

export default TopBar;
