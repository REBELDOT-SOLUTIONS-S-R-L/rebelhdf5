import { FiChevronsLeft, FiChevronsRight } from 'react-icons/fi';

import { useStore } from '../stores';
import sidebarStyles from './Sidebar.module.css';

interface Props {
  isCollapsed: boolean;
}

function SidebarToggle(props: Props) {
  const { isCollapsed } = props;
  const toggleSidebar = useStore((state) => state.toggleSidebar);

  return (
    <button
      className={sidebarStyles.navBtn}
      type="button"
      aria-label={`${isCollapsed ? 'Expand' : 'Collapse'} sidebar`}
      onClick={() => toggleSidebar()}
    >
      {isCollapsed ? <FiChevronsRight /> : <FiChevronsLeft />}
    </button>
  );
}

export default SidebarToggle;
