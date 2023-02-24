import { makeStyles } from 'tss-react/mui';
import flag from '/public/images/flag-logo.png';

const flagIcon = {
  width: 16,
  height: 16,
  borderRadius: '50%',
  display: 'inline-block',
  position: 'relative',
  marginRight: 5,
  top: 1,
  background: `url(${flag}) no-repeat transparent`,
  backgroundSize: '16px auto',
  '&[class="ar"]': {
    backgroundPosition: '0 3px'
  },
  '&[class="zh"]': {
    backgroundPosition: '0 -12px'
  },
  '&[class="en"]': {
    backgroundPosition: '0 -28px'
  },
  '&[class="de"]': {
    backgroundPosition: '0 -44px'
  },
  '&[class="id"]': {
    backgroundPosition: '0 -62px'
  },
  '&[class="pt"]': {
    backgroundPosition: '0 -79px'
  },
};

const settingsStyles = makeStyles({ uniqId: 'settings' })((theme, _params, classes) => ({
  active: {},
  langMenu: {
    textTransform: 'capitalize',
    '& i': {
      ...flagIcon
    }
  },
  icon: {
    '& svg': {
      fill: theme.palette.text.secondary
    }
  },
  setting: {
    [`& .${classes.icon}`]: {
      transition: 'all 0.3s ease',
    },
    [`& .${classes.active}`]: {
      transform: 'rotate(30deg)'
    }
  },
  modeMenu: {
    textTransform: 'capitalize'
  },
}));

// TODO jss-to-tss-react codemod: usages of this hook outside of this file will not be converted.
export default settingsStyles;
