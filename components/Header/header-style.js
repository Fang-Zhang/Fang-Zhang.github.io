import { makeStyles } from 'tss-react/mui';
import { alpha } from '@mui/material/styles';

const headerStyles = makeStyles({ uniqId: 'header' })((theme, _params, classes) => ({
  '@keyframes slideRight': {
    from: {
      opacity: 0,
      transform: 'translateX(-100px)'
    },
    to: {
      opacity: 1,
      transform: 'none'
    }
  },
  fixed: {},
  openDrawer: {},
  header: {
    color: theme.palette.text.primary,
    background: 'none',
    boxShadow: 'none',
    position: 'fixed',
    visibility: 'hidden',
    opacity: 0,
    zIndex: 90,
    [`&.${classes.fixed}`]: {
      transition: 'opacity 0.5s ease-out',
      position: 'fixed',
      visibility: 'visible',
      opacity: 1,
      background: alpha(theme.palette.background.paper, 0.8),
      backdropFilter: 'saturate(180%) blur(20px)',
      '& .MuiContainer-root': {
        padding: 0
      },
      [`& .${classes.logo}`]: {
        '& a': {
          color: theme.palette.text.primary,
        },
      },
      [`& .${classes.bar}`]: {
        [theme.breakpoints.down('md')]: {
          backgroundColor: theme.palette.text.secondary,
          '&:after, &:before': {
            backgroundColor: theme.palette.text.secondary
          }
        }
      }
    },
    [`&.${classes.openDrawer}`]: {
      background: 'none',
      backdropFilter: 'none'
    }
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    '& nav': {
      alignItems: 'center',
      padding: theme.spacing(0, 2),
      [theme.breakpoints.down('lg')]: {
        padding: theme.spacing(2, 0),
      },
      display: 'flex'
    }
  },
  logo: {
    flex: 1,
    display: 'flex',
    padding: theme.spacing(0, 2),
    justifyContent: 'center',
    '& a': {
      textDecoration: 'none',
      display: 'flex',
      fontSize: 22,
      color: theme.palette.text.primary,
      alignItems: 'center',
      fontWeight: 500,
      [theme.breakpoints.down('md')]: {
        color: theme.palette.common.white
      }
    },
    '& img': {
      transition: 'all 0.3s ease-out',
      width: 48,
      height: 48,
      marginRight: theme.spacing()
    }
  },
  invert: {},
  navLogo: {
    width: '100%',
    [`&.${classes.invert}`]: {
      [`& .${classes.logo}`]: {
        '& a': {
          color: theme.palette.text.primary,
        }
      },
      [`& .${classes.mobileMenu}`]: {
        [`& .${classes.bar}`]: {
          [theme.breakpoints.down('md')]: {
            backgroundColor: theme.palette.text.secondary,
            '&:after, &:before': {
              backgroundColor: theme.palette.text.secondary
            },
          }
        }
      },
    }
  },
  titleMenu: {
    textTransform: 'capitalize'
  },
  bar: {},
  menu: {},
  menuOpen: {},
  show: {},
  paperNav: {
    background: alpha(theme.palette.background.paper, 0.8),
    backdropFilter: 'saturate(180%) blur(20px)',
    width: '100%',
    position: 'fixed',
    zIndex: 50,
    height: '100%',
  },
  mobileMenu: {
    margin: theme.spacing(0, 1),
    [`& .${classes.bar}`]: {
      backgroundColor: theme.palette.text.secondary,
      '&:after, &:before': {
        backgroundColor: theme.palette.text.secondary
      },
      [theme.breakpoints.down('md')]: {
        backgroundColor: theme.palette.common.white,
        '&:after, &:before': {
          backgroundColor: theme.palette.common.white
        },
      }
    }
  },
  mobileNav: {
    [`& .${classes.menu}`]: {
      padding: theme.spacing(0, 2),
      overflow: 'auto',
      top: 80,
      width: '100%',
      position: 'absolute',
      height: 'calc(100% - 80px)',
      listStyle: 'none',
      '& li': {
        textAlign: 'center',
        animationName: '$slideRight',
        animationTimingFunction: 'ease',
        '& a': {
          fontSize: 24,
          padding: theme.spacing(1, 4),
          marginBottom: theme.spacing(),
          color: theme.palette.text.primary,
          '&:before': {
            content: '""',
            position: 'absolute',
            left: theme.spacing(-1),
            top: 5,
            opacity: 0.2,
            background: `linear-gradient(120deg, ${theme.palette.secondary.main}, ${theme.palette.primary.main})`,
            height: '75%',
            width: 0,
            transition: 'all 0.2s cubic-bezier(0, 0, 0.14, 0.97)'
          },
          '&:hover': {
            color: theme.palette.primary.main,
            background: 'none',
            '&:before': {
              width: '110%'
            },
          }
        }
      },
    }
  },
  menuList: {
    textTransform: 'capitalize',
    '& span': {
      fontSize: 24
    }
  }
}));

// TODO jss-to-tss-react codemod: usages of this hook outside of this file will not be converted.
export default headerStyles;
